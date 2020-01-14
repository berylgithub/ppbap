# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:41:10 2020

@author: Saint8312
"""

"""
functions to calculate the features of protein-protein interactions
"""

import numpy as np
import pandas as pd

'''
math functions
'''
f_euclid_dist = lambda a,b: np.linalg.norm(a-b)

def f_h_step(x, a):
    return 1 if (x<=a) else 0

f_y = lambda k : -np.log10(k)

'''
feature and labels calculator functions
'''
def y_processor(path):
    '''
    Create dataframes of log_10^y
    '''
    mol_units = {'uM':1.e-6, 'pM':1.e-12, 'fM':1.e-15, 'nM':1.e-9, 'mM':1.e-3}
    
    #load the index file
    l = []
    with open(path, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                l.append((line.rstrip()).split())
    df_idx = (pd.DataFrame(l)).rename(columns={0:'id',3:'k'})
    
    #generate the -log_10k values
    op_tokens = ['=','~','>','<']
    logys = np.zeros(df_idx.shape[0])
    for i in range(df_idx.shape[0]):
        string = df_idx.loc[i]['k']
        for s in string:
            if s in op_tokens:
                split_str = string.split(s)
                break
        logys[i] = f_y( float(split_str[-1][:-2]) * mol_units[split_str[-1][-2:]] )
    df_idx["log_y"] = logys
    return df_idx

def f_proteins_interaction(df_protein_A, df_protein_B, atom_types, cutoff):
    '''
    calculate the combination of euclidian distance and heaviside step between chains in a protein, 
    e.g chains=[A,B,C,D], hence the interactions are: [[A-B],[A-C],[A-D],[B-C],[B-D],[C-D]]
    
    'atom_types' are the type of atoms used for calculation
    'cutoff' is the distance cutoff between atoms for heaviside step function (in Angstrom)
    '''
    type_len = len(atom_types)
    x_vector = np.zeros(type_len**2)
    idx = 0
    for a_type in atom_types:
        for b_type in atom_types:
            #calculate the interaction of each atoms:
            sum_interaction = 0
            a_atoms = df_protein_A.loc[df_protein_A['atom_type'] == a_type]
            b_atoms = df_protein_B.loc[df_protein_B['atom_type'] == b_type]
            for i in range(a_atoms.shape[0]):
                for j in range(b_atoms.shape[0]):
                    #get the (x,y,z):
                    a_atom = a_atoms.iloc[i]
                    b_atom = b_atoms.iloc[j]
                    a_coord = np.array([float(a_atom['x_coor']), float(a_atom['y_coor']), float(a_atom['z_coor'])]) 
                    b_coord = np.array([float(b_atom['x_coor']), float(b_atom['y_coor']), float(b_atom['z_coor'])])
                    #calculate the euclidean distance and heaviside step value:
                    sum_interaction += f_h_step(x=f_euclid_dist(a_coord, b_coord), a=cutoff) 
            x_vector[idx] = sum_interaction
            idx+=1
            print(x_vector)
    return x_vector

def x_processor(chains, id_name, atom_types, cutoff):
    #vector calculation:
    x_vector = np.zeros(len(atom_types)**2)
    length = len(chains)
    for i in range(length):
        for j in range(length):
            if j>i:
                #sum each chain interaction values:
                print('protein chain :', i, j)
                x_vector += f_proteins_interaction(chains[i], chains[j], atom_types, cutoff)
    return {'id':id_name, 'x_vector':x_vector}

'''
multiprocessor feature functions
'''


if __name__ == '__main__':
    import pdb_processor as pdbp
    
    