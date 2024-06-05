# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 23:11:42 2024

@author: Alex
"""

from ase import Atoms
from ase.db import connect


db = connect('abc.db')

h2 = Atoms('H2', [(0, 0, 0), (0, 0, 0.7)])

db.write(h2, relaxed=False)