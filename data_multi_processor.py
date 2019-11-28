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
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP/index/INDEX_general_PP.2018'
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


if __name__ == '__main__':
    '''
    y data processor:
    '''
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP/index/INDEX_general_PP.2018'
    df_idx = y_data_processor(path)
    print(df_idx.loc[9])
    print(len(df_idx))
    
    '''
    files loader:
    '''
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
    
    complex_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    print(len(complex_files))
    
    test_file = path+'/'+complex_files[2]
    print(test_file)
    
    '''
    atom dataframe generator:
    '''
    l =[]
    with open(test_file, 'r') as f:
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
    df_atoms = (pd.DataFrame(l)).rename(columns={0:'record', 6:'x_coor', 7:'y_coor', 8:'z_coor', 11:'atom_type'})
    
    '''
    print(len(l[2314]))
    spl = [l[2314][-2][:4], l[2314][-2][4:]]
    l[2314][-2] = spl[1]
    l[2314].insert(-2, spl[0])
    l[2314].insert(-2, l[2314][-2][4:])
    print(l[2314])
    '''
    print(l[2241])
    print(df_atoms)