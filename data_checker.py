# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 17:56:59 2019

@author: Saint8312
"""

import pickle
import os
import numpy as np
import itertools

def data_load(filename):
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
    return data

    

if __name__ == '__main__':
#    data1 = data_load(os.getcwd()+'/Data/dataset_intel_1208190133.pkl')
#    saved_id1 = [d['id'] for d in data1]
#    print('processed protein IDs = ',saved_id1, print(len(saved_id1)))
#    
#    data2 = data_load(os.getcwd()+'/Data/dataset_ryzen_1208190143.pkl')
#    data2 = sorted(data2, key=lambda k: k['id']) 
#    saved_id2 = [d['id'] for d in data2]
#    print('processed protein IDs = ',saved_id2, print(len(saved_id2)))
#
#    comb_id = sorted(set(saved_id1+saved_id2))
#    print(comb_id, len(comb_id))
#    
#    comb_data = data1+data2
#    comb_data = list({d['id']:d for d in comb_data}.values())
#    print(comb_data[-1], data2[-1])
#    fname = 'dataset.pkl'
#    with open(fname, 'ab') as f:
#        for d in comb_data:
#            pickle.dump(d,f)
    
#    data1 = data_load('dataset.pkl')
#    saved_id1 = [d['id'] for d in data1]
#    print('processed protein IDs = ',saved_id1, print(len(saved_id1)))
    
    '''
    create subset matrices from the dataset, the default matrices should be (N,81) where N is the total data
    the subset will be (N, 16), taking only [C,N,O,S] atom types
    '''
    dataset = data_load(os.getcwd()+'/dataset.pkl')
    
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
    
    datalength = len(sorted_dataset)
    for i in range(datalength):
        sorted_dataset[i]['x_vector'] = sorted_dataset[i]['x_vector'][idx_l]
    
    ids = [d['id'] for d in sorted_dataset]
    print(ids)
    
    '''
    combine the standard atom interactions and hydrophobics & acids interactions vectors
    '''
    ha_dataset = data_load(os.getcwd()+'/h_a_vec.pkl')
    
    print( ha_dataset[0] )
    
    for i in range(datalength):
        new_vec = np.concatenate((sorted_dataset[i]['x_vector'], ha_dataset[i]['h_a_vector']))
        sorted_dataset[i]['x_vector'] = new_vec
        
    print(sorted_dataset[0], ha_dataset[0])
    
    '''
    save the combined dataset
    '''
    filename = os.getcwd()+'/Data/dataset_ha_alpha_122319.pkl'
    for i in range(datalength):
        with open(filename, 'ab') as f:
            pickle.dump(sorted_dataset[i], f)
    
    '''
    check the data
    '''
    dataset = data_load(filename)
    for d in dataset:
        print(d)
    print(len(dataset))
    