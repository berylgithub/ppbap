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
import itertools

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
            a_atoms = df_protein_A.loc[df_protein_A['element_symbol'] == a_type]
            b_atoms = df_protein_B.loc[df_protein_B['element_symbol'] == b_type]
            for i in range(a_atoms.shape[0]):
                for j in range(b_atoms.shape[0]):
                    #get the (x,y,z):
                    a_atom = a_atoms.iloc[i]
                    b_atom = b_atoms.iloc[j]
                    a_coord = np.array([float(a_atom['x_coord']), float(a_atom['y_coord']), float(a_atom['z_coord'])]) 
                    b_coord = np.array([float(b_atom['x_coord']), float(b_atom['y_coord']), float(b_atom['z_coord'])])
                    #calculate the euclidean distance and heaviside step value:
                    sum_interaction += f_h_step(x=f_euclid_dist(a_coord, b_coord), a=cutoff) 
            x_vector[idx] = sum_interaction
            idx+=1
    return x_vector

def x_atom_dist(chains, id_name, atom_types, cutoff):
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

def x_hydrophobic_acid(chains, id_name, cutoff):
    '''
    get the coordinates of CAs and then classify them based on the type of amino acid (hydrophobic, charged polar/acid), and then calculate the euclidean-heaviside as usual
    - hydrophobics = ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']
    - acids = ['ARG','HIS','LYS','ASP','GLU']
    '''

    print("processing ",id_name)
    
   
    #hydrophobics and acids types of amino acids
    hydrophobics = ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']
    acids = ['ARG','HIS','LYS','ASP','GLU']
    
    #select the carbon alpha of atoms based on the amino acid types
    hydrophobics_patches = []
    for i in range(len(chains)):
        mol_patch=chains[i].set_index(['residue_name'])
        hydrophobics_patches.append(mol_patch.loc[ (mol_patch.index.isin(hydrophobics)) & (mol_patch['atom_name'] == 'CA') ]) 
    
    acid_patches = []
    for i in range(len(chains)):
        mol_patch=chains[i].set_index(['residue_name'])
        acid_patches.append(mol_patch.loc[ (mol_patch.index.isin(acids)) & (mol_patch['atom_name'] == 'CA') ])
    
    patches = [hydrophobics_patches, acid_patches]
    
    #create the combination of protein patches interactions
    x_vector = np.zeros(2)
    patch_idx = 0
    for patch in patches:
        sum_interactions = 0
        comb_ = itertools.combinations(patch, 2)
        for c_ in list(comb_):
            #function to calculate the distance-cutoff between CAs of two protein patches:
            coors_0 = (c_[0][["x_coord", "y_coord", "z_coord"]]).to_numpy(dtype=float)
            coors_1 = (c_[1][["x_coord", "y_coord", "z_coord"]]).to_numpy(dtype=float)
            product_coors = np.array(list(itertools.product(coors_0, coors_1)))
#                if pool:
#                euclid_dists = pool.map(data_multi_processor.f_euc_mp, product_coors)
#                euclid_dists = np.array(list(euclid_dists))
#                paramlist = list(itertools.product(euclid_dists, [cutoff]))
#                heavisides = pool.map(data_multi_processor.f_heaviside_mp, paramlist)
#                heavisides = np.array(list(heavisides))
#                else:
            euclid_dists = np.array(list(map(f_euc_mp, product_coors)))
            paramlist = list(itertools.product(euclid_dists, [cutoff]))
            heavisides = np.array(list(map(f_heaviside_mp, paramlist)))
            sum_interactions += np.sum(heavisides)
            x_vector[patch_idx] = sum_interactions
        patch_idx+=1
    return {'id':id_name, 'h_a_vector':x_vector}

def x_processor(chains, id_name, atom_types, cutoff):
    x_dict = x_atom_dist(chains, id_name, atom_types, cutoff)
    ha_dict = x_hydrophobic_acid(chains, id_name, cutoff)
    x_dict['x_vector'] = np.concatenate((x_dict['x_vector'], ha_dict["h_a_vector"]))
    return x_dict

'''
multiprocessor feature functions
'''
def f_euc_mp(params):
    return np.linalg.norm(params[0]-params[1])

def f_heaviside_mp(params):
    return 1 if(params[0]<=params[1]) else 0


def f_proteins_interaction_mp(df_protein_A, df_protein_B, atom_types, cutoff, pool):
    type_len = len(atom_types)
    x_vector = np.zeros(type_len**2)
    idx = 0
    for a_type in atom_types:
        for b_type in atom_types:
            #calculate the interaction of each atoms:
            sum_interaction = 0
            a_atoms = df_protein_A.loc[df_protein_A['element_symbol'] == a_type].to_dict('records')
            b_atoms = df_protein_B.loc[df_protein_B['element_symbol'] == b_type].to_dict('records')
            a_coords = np.array([[a_atom['x_coord'], a_atom['y_coord'], a_atom['z_coord']] for a_atom in a_atoms], dtype=float)
            b_coords = np.array([[b_atom['x_coord'], b_atom['y_coord'], b_atom['z_coord']] for b_atom in b_atoms], dtype=float) 
            paramlist = list(itertools.product(a_coords, b_coords))
            euclid_dists = pool.map(f_euc_mp, paramlist)
            euclid_dists = np.array(list(euclid_dists))            
            paramlist = list(itertools.product(euclid_dists, [cutoff]))
            heavisides = pool.map(f_heaviside_mp, paramlist)
            heavisides = np.array(list(heavisides))
            sum_interaction = np.sum(heavisides)
            x_vector[idx] = sum_interaction
            idx+=1
    return x_vector

def x_atom_dist_mp(params):
    chains = params[0]
    id_name = params[1]
    atom_types = params[2]
    cutoff = params[3]
    pool = params[4]
    #vector calculation:
    x_vector = np.zeros(len(atom_types)**2)
    length = len(chains)
    for i in range(length):
        for j in range(length):
            if j>i:
                #sum each chain interaction values:
                print('protein chain :', i, j)
                x_vector += f_proteins_interaction_mp(chains[i], chains[j], atom_types, cutoff, pool)
    return {'id':id_name, 'x_vector':x_vector}

def x_hydrophobic_acid_mp(params):
    chains = params[0]
    id_name = params[1]
    cutoff = params[2]
    pool = params[3]
    
    print("processing ",id_name)
    
    #hydrophobics and acids types of amino acids
    hydrophobics = ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']
    acids = ['ARG','HIS','LYS','ASP','GLU']
    
    #select the carbon alpha of atoms based on the amino acid types
    hydrophobics_patches = []
    for i in range(len(chains)):
        mol_patch=chains[i].set_index(['residue_name'])
        hydrophobics_patches.append(mol_patch.loc[ (mol_patch.index.isin(hydrophobics)) & (mol_patch['atom_name'] == 'CA') ]) 
    
    acid_patches = []
    for i in range(len(chains)):
        mol_patch=chains[i].set_index(['residue_name'])
        acid_patches.append(mol_patch.loc[ (mol_patch.index.isin(acids)) & (mol_patch['atom_name'] == 'CA') ])
    
    patches = [hydrophobics_patches, acid_patches]
    
    #create the combination of protein patches interactions
    x_vector = np.zeros(2)
    patch_idx = 0
    for patch in patches:
        sum_interactions = 0
        comb_ = itertools.combinations(patch, 2)
        for c_ in list(comb_):
            #function to calculate the distance-cutoff between CAs of two protein patches:
            coors_0 = (c_[0][["x_coord", "y_coord", "z_coord"]]).to_numpy(dtype=float)
            coors_1 = (c_[1][["x_coord", "y_coord", "z_coord"]]).to_numpy(dtype=float)
            product_coors = np.array(list(itertools.product(coors_0, coors_1)))
            euclid_dists = pool.map(f_euc_mp, product_coors)
            euclid_dists = np.array(list(euclid_dists))
            paramlist = list(itertools.product(euclid_dists, [cutoff]))
            heavisides = pool.map(f_heaviside_mp, paramlist)
            heavisides = np.array(list(heavisides))
#            euclid_dists = np.array(list(map(f_euc_mp, product_coors)))
#            paramlist = list(itertools.product(euclid_dists, [cutoff]))
#            heavisides = np.array(list(map(f_heaviside_mp, paramlist)))
            sum_interactions += np.sum(heavisides)
            x_vector[patch_idx] = sum_interactions
        patch_idx+=1
    return {'id':id_name, 'h_a_vector':x_vector}

def x_processor_mp(params):
    chains = params[0]
    id_name = params[1]
    atom_types = params[2]
    cutoff = params[3]
    pool = params[4]
    x_dict = x_atom_dist([chains, id_name, atom_types, cutoff, pool])
    ha_dict = x_hydrophobic_acid([chains, id_name, cutoff, pool])
    x_dict['x_vector'] = np.concatenate((x_dict['x_vector'], ha_dict["h_a_vector"]))
    return x_dict
