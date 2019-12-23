# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:19:00 2019

@author: Saint8312
"""
import numpy as np
import pandas as pd
import sys, os
import time
import multiprocessing
import itertools
import pickle
import data_checker
import data_multi_processor





def hydrophobic_acid_patch_interactions(paramlist):
    '''
    get the coordinates of CAs and then classify them based on the type of amino acid (hydrophobic, charged polar/acid), and then calculate the euclidean-heaviside as usual
    - hydrophobics = ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']
    - acids = ['ARG','HIS','LYS','ASP','GLU']
    '''
    path = paramlist[0]
    id_name = paramlist[1]
    cutoff = paramlist[2]
    print("processing ",id_name)
    
    path_file = path+'/'+id_name
    l =[]
    with open(path_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                clean_line = (line.rstrip()).split()
                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
                if len(clean_line) == 11:
                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
                    split = [clean_line[-2][:4], clean_line[-2][4:]]
                    clean_line[-2] = split[1]
                    clean_line.insert(-2, split[0])
                #check if the chain identifier is misaligned
                if len(clean_line[4])>1:
                    split = [clean_line[4][0], clean_line[4][1:]]
                    clean_line[4] = split[0]
                    clean_line.insert(5, split[1])
                #check if coordinate data collumns are collided (most likely happens between x and y coor)
                while len(clean_line[6])>=13:
                    split = [clean_line[6][:-8], clean_line[6][-8:]]
                    last_elem = clean_line.pop()
                    clean_line[-1] = last_elem
                    clean_line.insert(6, split[0])
                    clean_line[7] = split[1]
                if len(clean_line[7])>=13:
                    split = [clean_line[7][:-8], clean_line[7][-8:]]
                    last_elem = clean_line.pop()
                    clean_line[-1] = last_elem
                    clean_line.insert(7, split[0])
                    clean_line[8] = split[1]
                l.append(clean_line)
            elif line.startswith('TER'):
                clean_line = (line.rstrip()).split()
                l.append(clean_line)
            elif line.startswith('ENDMDL'):
                break
    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type', 2:'atom_name', 3:'amino_acid_type'})
    
    #dataframe splitter:
    l_df = []
    last_idx = 0
    for idx in df_atoms.index[df_atoms['record'] == 'TER'].tolist():
        l_df.append(df_atoms.iloc[last_idx:idx])
        last_idx = idx+1
    
    #hydrophobics and acids types of amino acids
    hydrophobics = ['ALA','VAL','ILE','LEU','MET','PHE','TYR','TRP']
    acids = ['ARG','HIS','LYS','ASP','GLU']
    
    #select the carbon alpha of atoms based on the amino acid types
    hydrophobics_patches = []
    for i in range(len(l_df)):
        mol_patch=l_df[i].set_index(['amino_acid_type'])
        hydrophobics_patches.append(mol_patch.loc[ (mol_patch.index.isin(hydrophobics)) & (mol_patch['atom_name'] == 'CA') ]) 
    
    acid_patches = []
    for i in range(len(l_df)):
        mol_patch=l_df[i].set_index(['amino_acid_type'])
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
            coors_0 = (c_[0][["x_coor", "y_coor", "z_coor"]]).to_numpy(dtype=float)
            coors_1 = (c_[1][["x_coor", "y_coor", "z_coor"]]).to_numpy(dtype=float)
            product_coors = np.array(list(itertools.product(coors_0, coors_1)))
#                if pool:
#                euclid_dists = pool.map(data_multi_processor.f_euc_mp, product_coors)
#                euclid_dists = np.array(list(euclid_dists))
#                paramlist = list(itertools.product(euclid_dists, [cutoff]))
#                heavisides = pool.map(data_multi_processor.f_heaviside_mp, paramlist)
#                heavisides = np.array(list(heavisides))
#                else:
            euclid_dists = np.array(list(map(data_multi_processor.f_euc_mp, product_coors)))
            paramlist = list(itertools.product(euclid_dists, [cutoff]))
            heavisides = np.array(list(map(data_multi_processor.f_heaviside_mp, paramlist)))
            sum_interactions += np.sum(heavisides)
            x_vector[patch_idx] = sum_interactions
        patch_idx+=1
    return {'id':id_name, 'h_a_vector':x_vector}


if __name__ == '__main__':
    '''
    create subset matrices from the dataset, the default matrices should be (N,81) where N is the total data
    the subset will be (N, 16), taking only [C,N,O,S] atom types
    '''
    dataset = data_checker.data_load(os.getcwd()+'/dataset.pkl')
    saved_id = [d['id'] for d in dataset]
#    print('processed protein IDs = ',saved_id, print(len(saved_id)))
    
    sorted_dataset = sorted(dataset, key = lambda k:k['id'])
    atom_types = ['C','N','O','F','P','S','Cl','Br','I']
    subset_atom_types = ['C','N','O','S']
    subset_exclude = list(set(atom_types)-set(subset_atom_types))
    paramlist = list(itertools.product(atom_types, atom_types))
#    print(paramlist)
    idx_l = []
    for i in range(len(paramlist)):
        result =  not any(elem in paramlist[i] for elem in subset_exclude)
        if result:
            idx_l.append(i)
    
    for i in range(len(sorted_dataset)):
        sorted_dataset[i]['x_vector'] = sorted_dataset[i]['x_vector'][idx_l]
    
    '''
    hydrophobic and acid patches interactions calculation
    ''' 
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
    complex_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    complexes = complex_files
    cutoff = 12
#    
    filename = 'h_a_vec.pkl'

#    pool = multiprocessing.Pool()
    
    #start of the process
    start_time = time.time()
        
#    paramlist = list(itertools.product([path], complexes, [cutoff]))
#    h_a_vec = pool.map(hydrophobic_acid_patch_interactions, paramlist)
#    h_a_vec = list(h_a_vec)
    
    #check if id is already existed within file, if yes, skip it
    data = []
    try:
        with open(filename, 'rb') as fr:
            print(filename, 'is found')
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass            
    except FileNotFoundError:
        print('File is not found')
    saved_ids = [d['id'] for d in data]
    
    try:
        i=0
        for id_file in complexes:
            if id_file in saved_ids:
                continue
            else:
                print("start of process for ID :",id_file)
                paramlist = [path, id_file, cutoff]
                h_a_vec = hydrophobic_acid_patch_interactions(paramlist)
                print("ID : ", id_file)
                print('value of x vector (R^N) = ', h_a_vec)
                with open(filename, 'ab') as f:
                    pickle.dump(h_a_vec, f)
                i+=1
    except KeyboardInterrupt:
        print('interrupted !!')
        

    end_time = time.time()
    print("the number of protein processed in current run = ",i)
    print('time elapsed =',end_time-start_time,'seconds')
    
    '''
    data checker
    '''
    data = data_checker.data_load(filename)
    print(data, len(data))
    
#    id_name = '3s5l.ent.pdb'
#    path_file = path+'/'+id_name
#    l =[]
#    with open(path_file, 'r') as f:
#        for line in f:
#            if line.startswith('ATOM'):
#                clean_line = (line.rstrip()).split()
#                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
#                if len(clean_line) == 11:
#                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
#                    split = [clean_line[-2][:4], clean_line[-2][4:]]
#                    clean_line[-2] = split[1]
#                    clean_line.insert(-2, split[0])
#                #check if the chain identifier is misaligned
#                if len(clean_line[4])>1:
#                    split = [clean_line[4][0], clean_line[4][1:]]
#                    clean_line[4] = split[0]
#                    clean_line.insert(5, split[1])
#                #check if coordinate data collumns are collided (most likely happens between x and y coor)
#                while len(clean_line[6])>=13:
#                    split = [clean_line[6][:-8], clean_line[6][-8:]]
#                    last_elem = clean_line.pop()
#                    clean_line[-1] = last_elem
#                    clean_line.insert(6, split[0])
#                    clean_line[7] = split[1]
#                if len(clean_line[7])>=13:
#                    split = [clean_line[7][:-8], clean_line[7][-8:]]
#                    last_elem = clean_line.pop()
#                    clean_line[-1] = last_elem
#                    clean_line.insert(7, split[0])
#                    clean_line[8] = split[1]
#                l.append(clean_line)
#            elif line.startswith('TER'):
#                clean_line = (line.rstrip()).split()
#                l.append(clean_line)
#            elif line.startswith('ENDMDL'):
#                break
