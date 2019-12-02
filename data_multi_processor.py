# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 17:32:38 2019

@author: Saint8312
"""

import numpy as np
import pandas as pd
import sys, os
import time
import multiprocessing
import itertools
import pickle

'''
math functions
'''
f_euclid_dist = lambda a,b: np.linalg.norm(a-b)

def f_h_step(x, a):
    return 1 if (x<=a) else 0

f_y = lambda k : -np.log10(k)


def y_data_processor(path):
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

def protein_interaction(df_protein_A, df_protein_B, atom_types, cutoff):
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

def data_processing(path,id_name, atom_types, cutoff):
    #dataframe loader:
    path_file = path+'/'+id_name
    l =[]
    with open(path_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('TER'):
                clean_line = (line.rstrip()).split()
                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
                if len(clean_line) == 11:
                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
                    split = [clean_line[-2][:4], clean_line[-2][4:]]
                    clean_line[-2] = split[1]
                    clean_line.insert(-2, split[0])
                l.append(clean_line)
            elif line.startswith('ENDMDL'):
                break
    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type'})
    
    #dataframe splitter:
    l_df = []
    last_idx = 0
    for idx in df_atoms.index[df_atoms['record'] == 'TER'].tolist():
        l_df.append(df_atoms.iloc[last_idx:idx])
        last_idx = idx+1
        
    #vector calculation:
    x_vector = np.zeros(len(atom_types)**2)
    length = len(l_df)
    for i in range(length):
        for j in range(length):
            if j>i:
                #sum each chain interaction values:
                print('protein chain :', i, j)
                x_vector += protein_interaction(l_df[i], l_df[j], atom_types, cutoff)
    return {'id':id_name, 'x_vector':x_vector}


###########################################
'''
multiprocessing functions
'''
def f_euc_mp(params):
    return np.linalg.norm(params[0]-params[1])

def f_heaviside_mp(params):
    return 1 if(params[0]<=params[1]) else 0

def protein_interaction_mp(df_protein_A, df_protein_B, atom_types, cutoff, pool):
    type_len = len(atom_types)
    x_vector = np.zeros(type_len**2)
    idx = 0
    for a_type in atom_types:
        for b_type in atom_types:
            #calculate the interaction of each atoms:
            sum_interaction = 0
            a_atoms = df_protein_A.loc[df_protein_A['atom_type'] == a_type].to_dict('records')
            b_atoms = df_protein_B.loc[df_protein_B['atom_type'] == b_type].to_dict('records')
            a_coords = np.array([[a_atom['x_coor'], a_atom['y_coor'], a_atom['z_coor']] for a_atom in a_atoms], dtype=float)
            b_coords = np.array([[b_atom['x_coor'], b_atom['y_coor'], b_atom['z_coor']] for b_atom in b_atoms], dtype=float) 
            paramlist = list(itertools.product(a_coords, b_coords))
            euclid_dists = pool.map(f_euc_mp, paramlist)
            euclid_dists = np.array(list(euclid_dists))            
            paramlist = list(itertools.product(euclid_dists, [cutoff]))
            heavisides = pool.map(f_heaviside_mp, paramlist)
            heavisides = np.array(list(heavisides))
            sum_interaction = np.sum(heavisides)
            x_vector[idx] = sum_interaction
            idx+=1
            print(x_vector)
    return x_vector

def data_multi_processing(path,id_name, atom_types, cutoff, pool):
    #dataframe loader:
    path_file = path+'/'+id_name
    l =[]
    with open(path_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('TER'):
                clean_line = (line.rstrip()).split()
                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
                if len(clean_line) == 11:
                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
                    split = [clean_line[-2][:4], clean_line[-2][4:]]
                    clean_line[-2] = split[1]
                    clean_line.insert(-2, split[0])
                l.append(clean_line)
            elif line.startswith('ENDMDL'):
                break
    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type'})
    
    #dataframe splitter:
    l_df = []
    last_idx = 0
    for idx in df_atoms.index[df_atoms['record'] == 'TER'].tolist():
        l_df.append(df_atoms.iloc[last_idx:idx])
        last_idx = idx+1
        
    #vector calculation:
    x_vector = np.zeros(len(atom_types)**2)
    length = len(l_df)
    for i in range(length):
        for j in range(length):
            if j>i:
                #sum each chain interaction values:
                print('protein chain :', i, j)
                x_vector += protein_interaction_mp(l_df[i], l_df[j], atom_types, cutoff, pool)
    return {'id':id_name, 'x_vector':x_vector}

def data_multi_processing_mp(params):
    #dataframe loader:
    path_file = params[0]+'/'+params[1]
    l =[]
    with open(path_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('TER'):
                clean_line = (line.rstrip()).split()
                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
                if len(clean_line) == 11:
                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
                    split = [clean_line[-2][:4], clean_line[-2][4:]]
                    clean_line[-2] = split[1]
                    clean_line.insert(-2, split[0])
                l.append(clean_line)
            elif line.startswith('ENDMDL'):
                break
    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type'})
    
    #dataframe splitter:
    l_df = []
    last_idx = 0
    for idx in df_atoms.index[df_atoms['record'] == 'TER'].tolist():
        l_df.append(df_atoms.iloc[last_idx:idx])
        last_idx = idx+1
        
    #vector calculation:
    x_vector = np.zeros(len(params[2])**2)
    length = len(l_df)
    for i in range(length):
        for j in range(length):
            if j>i:
                #sum each chain interaction values:
                print('protein chain :', i, j)
                x_vector += protein_interaction_mp(l_df[i], l_df[j], params[2], params[3], params[4])
    return {'id':params[1], 'x_vector':x_vector}

if __name__ == '__main__':
    
    def unit_test_data_processing():
        '''
        data processing unit test
        '''
        path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
        id_file = complex_files[2]
        atom_types = ['C','N','O','F','P','S','Cl','Br','I']
        cutoff = 12
        
        curr_time = time.time()
        x_vector = data_processing(path, id_file, atom_types, cutoff)
        print('value of x vector (R^N) = ', x_vector)
        end_time = time.time()
        print('time elapsed =',end_time-curr_time,'seconds')
        
    
    
    def unit_test_y_data():
        '''
        y data processor:
        '''
        path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP/index/INDEX_general_PP.2018'
        df_idx = y_data_processor(path)
        print(df_idx.loc[9])
        print(len(df_idx))
        
        

#    '''
#    files loader:
#    '''
#    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
#    
#    complex_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
#    print(len(complex_files))
#    
##    test_file = path+'/'+complex_files[2]
#    test_file = path+'/2wy2.ent.pdb'
#    print(test_file)
#    
#    '''
#    atom dataframe generator:
#    '''
#    l =[]
#    with open(test_file, 'r') as f:
#        for line in f:
#            if line.startswith('ATOM') or line.startswith('TER'):
#                clean_line = (line.rstrip()).split()
#                #check for alignment mistakes within data, a row with spacing alignment error has 11 length after splitted by whitespace
#                if len(clean_line) == 11:
#                    #split the 2nd last column by the 4th index (this inference is according to PDB file formatting)
#                    split = [clean_line[-2][:4], clean_line[-2][4:]]
#                    clean_line[-2] = split[1]
#                    clean_line.insert(-2, split[0])
#                l.append(clean_line)
#            elif line.startswith('ENDMDL'):
#                break
#    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type'})
#    
#
#    print(df_atoms)
#    
#    '''
#    split dataframes based on chains ended by "TER"
#    '''
#    l_df = []
#    last_idx = 0
#    for idx in df_atoms.index[df_atoms['record'] == 'TER'].tolist():
#        l_df.append(df_atoms.iloc[last_idx:idx])
#        last_idx = idx+1
#    
#    print(df_atoms.index[df_atoms['record'] == 'TER'].tolist())
#    print(l_df)
    
    '''
    multiprocessing unit test
    '''
#    #parameters:
#    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
#    atom_types = ['C','N','O','F','P','S','Cl','Br','I']
#    cutoff = 12
#    id_file = '2wy2.ent.pdb'
#    complexes = complex_files[0:3]
# 
#    #process:
#    start_time = time.time()
#    pool = multiprocessing.Pool()
#    
#    x_vector = data_multi_processing(path, id_file, atom_types, cutoff, pool)
#    print('value of x vector (R^N) = ', x_vector)

    '''
    using map
    '''
#    paramlist = list(itertools.product([path], complex_files, [atom_types], [cutoff], [pool]))
#    sample_params = paramlist[0:3]
#    print(sample_params)
#    x_vector = map(data_multi_processing_mp, sample_params)
#    x_vector = np.array(list(x_vector))
#    print('value of x vector (R^N) = ', x_vector)
    
    '''
    using for-loop
    '''
#    for id_file in sample_complex:
#        x_vector = data_multi_processing(path, id_file, atom_types, cutoff, pool)
#        print('value of x vector (R^N) = ', x_vector)
#        with open(filename, 'ab') as f:
#            pickle.dump(x_vector, f)
            
    
#    end_time = time.time()
#    print('time elapsed =',end_time-start_time,'seconds')


    '''
    data processing & writing
    '''
    #initialize parameters
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
    complex_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
    atom_types = ['C','N','O','F','P','S','Cl','Br','I']
    cutoff = 12
    complexes = complex_files
    filename = "dataset.pkl"
    
    #start of the process
    start_time = time.time()
    pool = multiprocessing.Pool()
    
    #y_data loader
    df_y = y_data_processor('C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP/index/INDEX_general_PP.2018')

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

    #process and save the data
    try:
        i=0
        for id_file in complexes:
            if id_file in saved_ids:
                continue
            else:
                vector = data_multi_processing(path, id_file, atom_types, cutoff, pool)
                y = df_y.loc[df_y['id']==id_file.split('.')[0]]['log_y'].values[0]
                vector["y"]=y
                print("ID : ", id_file)
                print('value of x vector (R^N) = ', vector)
                with open(filename, 'ab') as f:
                    pickle.dump(vector, f)
                i+=1
    except KeyboardInterrupt:
        print('interrupted !!')
    
    end_time = time.time()
    print("the number of protein processed in current run = ",i)
    print('time elapsed =',end_time-start_time,'seconds')
            
    
    '''
    data checker
    '''
    data = []
    try:
        with open(filename, 'rb') as fr:
            try:
                while True:
                    data.append(pickle.load(fr))
            except EOFError:
                pass            
    except FileNotFoundError:
        print('File is not found')
    saved_ids = [d['id'] for d in data]
    print('processed protein IDs = ',saved_ids)
    