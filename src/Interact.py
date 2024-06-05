# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 15:37:47 2024

@author: Alex
"""

from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()

dbname          = "./database_q.db"   # name of the database used throughout the training (*.json or *.db file formats are allowed)
elements        =  [1,8]              # possible atomic numbers of the elements found in the database (e.g H,O)

os.environ['frset']   = os.pathsep.join(str(i) for i in sorted(elements))

import SchNet4AIM
import torch
import json
from SchNet4AIM import AtomsLoader as s4aim_AtomsLoader
from SchNet4AIM import create_subset as s4aim_create_subset


with open("SchNet4AIM/examples/databases/energetic.json") as jin:
#with open("SchNet4AIM/examples/databases/electronic.json") as jin:
    data = json.load(jin)

for key in data:
    print(key)
    print(data[key][0])

def read_from_json(databasename):
    """
    A function to read databases stored in a JSON-like datafile.
    """
    with open(databasename) as jFile:
        jObject = json.load(jFile)
    jFile.close()
    return jObject

def load_json_database_as_ase(dbname):
    """
    A function to load a json database as an ASE dataset.
    """
    dataset = read_from_json(dbname)
    dataset_ase = SchNet4AIM.AtomsData(dbname)
    atoms, properties = dataset_ase.get_properties(0)
    return dataset_ase,atoms,properties

#dbname = "SchNet4AIM/examples/databases/energetic.json"
#dbname = "SchNet4AIM/examples/databases/electronic.json"
#dataset_ase,atoms,properties = load_json_database_as_ase(dbname)

#dataset = SchNet4AIM.AtomsLoader("SchNet4AIM/examples/databases/energetic.json")

dbname = "abc.db"
dataset_ase = SchNet4AIM.AtomsData(dbname)

subset = s4aim_create_subset(dataset_ase, np.arange(0,len(dataset_ase)))

x = s4aim_AtomsLoader(subset,batch_size=1,shuffle=False)

model = torch.load("SchNet4AIM/examples/models/energetic/model_inter_AIMwise", map_location=torch.device('cpu'))

for count, batch in enumerate(x.dataset):
    print(batch)

    pred = model(batch)
    break

