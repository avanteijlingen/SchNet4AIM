# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:24:19 2024

@author: Alex
"""
import torch, ase, tqdm
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv
import os, json, sys
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

sys.path.append("C:/Users/Alex/Documents/GitHub/EquivariantGNN/")
from egnn.model.convolution import GraphConvolution, EquivariantGraphConvolution
from egnn.model.gnn import GNN, EquivariantGNN
from data import *

device = torch.device("cpu")

#enc = OneHotEncoder(handle_unknown='ignore') # ignore will mean it sets atomic number '0' to [0,0,0,0 .... 0]
enc = OneHotEncoder(handle_unknown='error') # ignore will mean it sets atomic number '0' to [0,0,0,0 .... 0]

enc.fit(np.array(["H", "C", "N", "O"]).reshape(-1, 1))




class orbital_dataset(Dataset):
    def __init__(self, jsonfile):
        with open(jsonfile) as jin:
            self.data = json.load(jin)
        print(self.data.keys())
        self.max_num_atoms = 100

    def __len__(self):
        return len(self.data["natoms"])
    

    def __getitem__(self, idx):
        num_atoms = self.data["natoms"][idx]
        positions = torch.tensor(self.data["pos"][idx])
        elements = np.array(self.data["ele"][idx])
        one_hot = torch.from_numpy(enc.transform(elements.reshape(-1,1)).toarray().astype(np.int32))
        charges = torch.tensor([ase.data.atomic_numbers[x] for x in elements])
        edge_features = torch.cdist(positions, positions).float()
        atomic_charges = torch.tensor(self.data["q"][idx])
        
        #edge_features = edge_features.reshape(-1)
        # Apply padding
# =============================================================================
#         padding = self.max_num_atoms - positions.shape[0]
#         #edge_features = torch.nn.functional.pad(edge_features, (0,padding,0,padding))
#         #edge_features = torch.nn.functional.pad(edge_features, (0,padding,0,0))
#         #print("edge_features.size()", edge_features.size())
#         
#         #print("charges", charges.shape)
#         charges = torch.nn.functional.pad(charges, (0,padding))
#         atomic_charges = torch.nn.functional.pad(atomic_charges, (0,padding))
#         #print("charges", charges.shape)
#         positions = torch.nn.functional.pad(positions, (0,0,0,padding))
#         edge_features = torch.nn.functional.pad(edge_features, (0,padding,0,padding))
#         #print("one_hot", one_hot.shape)
#         one_hot = torch.nn.functional.pad(one_hot, (0, 0, 0, padding))
# =============================================================================
# =============================================================================
#         print("one_hot", one_hot.shape)
#         print("padding", padding)
# =============================================================================

        edge_ids = torch.vstack((torch.where((edge_features < 15) * (edge_features > 0.001)))).T
        edge_features = edge_features[torch.where((edge_features < 15) * (edge_features > 0.001))]
        
        
        data = {"num_atoms": torch.tensor(num_atoms),
                #"energy": Energy,
                "charges": charges,
                "positions": positions,
                "edge_features": edge_features,
                "one_hot": one_hot,
                "atomic_charges": atomic_charges,
                "edge_ids": edge_ids
                }
        return data
    
dset = orbital_dataset("SchNet4AIM/examples/databases/electronic.json")

print(len(dset))


collator = collation(device, cutoff = 15.0)

training_set = DataLoader(dset,
                        batch_size = 1,
                        shuffle = True,
                        pin_memory = False,
                        collate_fn = collator.collate_fn, # processes the data on load
                        )

model = EquivariantGNN(    
    input_node_dim = 7,
    input_edge_dim = 1,
    #output_dim = ds.curr_ds, # for testing multi output training
    output_dim = 1, # for testing multi output training
    #output_dim = 1,
    attention = False, 
    num_coords = 3,
    num_velocities = 0, 
    update_coords = False, # for testing multi output training
    hidden_dim = 64,
    num_layers = 3,
    #predict_type = "node_pooling",
    predict_type = "unembedding"
    #predict_type = "coord"
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr = 0.001,  weight_decay =0.99) # it is 1 per local_rank which is attached to the model
#optimizer = torch.optim.SGD(model.parameters(), lr = 0.000001, weight_decay =0.9) # it is 1 per local_rank which is attached to the model
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10, threshold=0.0, verbose=True)

mse = torch.nn.MSELoss(reduction='mean')


batch_size = 2**10

positions = None
for epoch in range(10):
    local_batch_size = 0
    epoch_loss = 0 
    nbatches = 0
    
    for idx in tqdm.tqdm(torch.randperm(len(dset))):
        
        sample = dset.__getitem__(idx)
        
        if positions is None:
            #Stack the up
            positions = sample["positions"]
            one_hot = sample["one_hot"]
            atomic_charges = sample["atomic_charges"]
            
            edge_features = sample["edge_features"].reshape(-1,1)
            edge_ids = sample["edge_ids"]
            
            n = positions.shape[0]
            local_batch_size += 1
            
        elif local_batch_size <= batch_size:
            positions = torch.vstack((positions, sample["positions"]))
            one_hot = torch.vstack((one_hot, sample["one_hot"]))
            atomic_charges = torch.hstack((atomic_charges, sample["atomic_charges"]))
            
            edge_ids = torch.vstack(( edge_ids, sample["edge_ids"] + n ))
            edge_features = torch.vstack((edge_features, sample["edge_features"].reshape(-1,1)))
            
            n += sample["positions"].shape[0]
            local_batch_size += 1
            
        if local_batch_size < batch_size:
            continue


        nodes = torch.hstack((positions, one_hot)).float()
    
        out, new_edge_indices, new_edges_features = model(nodes, edge_ids.T, edge_features, batch_size = batch_size)
        out = out.flatten()
    
        loss = mse(out, atomic_charges)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        AdamW_scheduler.step(loss)

        epoch_loss += loss.item()
        nbatches += 1
        positions = None
        local_batch_size = 0
        
        
# =============================================================================
#         USE THE ATOMIC CHARGES AS INPUTS (ADDITIONAL TO THE NODES) IN ENERGY PREDICTION
#         BASICALLY LIKE DOING 'update_coords' BUT WITH ATOMIC CHARGES
# =============================================================================
        
    epoch_loss /= nbatches 
    print("Epoch:", epoch, "Loss:", epoch_loss)
    

