#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:15:20 2020

@author: saint8312
"""

import pdb_processor as pdbp
import feature_calculator as fc
import time
from multiprocessing import Pool
import pickle





if __name__=="__main__":
    import json

    with open('config.json') as json_data_file:
        conf = json.load(json_data_file)
        
    x_path = conf['root']['PP']
    y_path = conf['index']['PP']    
    complex_files = pdbp.list_files(x_path)
    
    atom_types = ['C','N','O','S']
    cutoff = 12
    
    start_time = time.time()
    pool = Pool(6)    
    



#    '''
#    dataset generator
#    '''
#    filename = "dataset_beta.pkl"
#    #y_data loader
#    df_y = fc.y_processor(conf['index']['PP'])
#
#    #check if id is already existed within file, if yes, skip it
#    data = []
#    try:
#        with open(filename, 'rb') as fr:
#            print(filename, 'is found')
#            try:
#                while True:
#                    data.append(pickle.load(fr))
#            except EOFError:
#                pass            
#    except FileNotFoundError:
#        print('File is not found')
#    saved_ids = [d['id'] for d in data]
#
#    #process and save the data
#    try:
#        i=0
#        for id_name in complex_files:
#            if id_name in saved_ids:
#                continue
#            else:
#                print("start of process for ID :",id_name)
#                pathfile = x_path+"/"+id_name
#                chains = pdbp.loader_pdbbind(pathfile)    
#                vector = fc.x_processor_mp([chains, id_name, atom_types, cutoff, pool])
#                y = df_y.loc[df_y['id']==id_name.split('.')[0]]['log_y'].values[0]
#                vector["y"]=y
#                print("ID : ", id_name)
#                print('value of x vector (R^N) = ', vector)
#                with open(filename, 'ab') as f:
#                    pickle.dump(vector, f)
#                i+=1
#    except KeyboardInterrupt:
#        print('interrupted !!')
#    
#    end_time = time.time()
#    print("the number of protein processed in current run = ",i)
#    print('time elapsed =',end_time-start_time,'seconds')
    
    '''
    inference data
    '''
    x_path = conf['root']['zdock']["4AZU"]
    complex_files = pdbp.list_files(x_path)
    print(complex_files)
    filename = "Data/data_4AZU.pkl"
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
        for id_name in complex_files:
            if id_name in saved_ids:
                continue
            else:
                print("start of process for ID :",id_name)
                pathfile = x_path+"/"+id_name
                chains = pdbp.loader_pdbbind(pathfile)    
                vector = fc.x_processor_mp([chains, id_name, atom_types, cutoff, pool])
                print("ID : ", id_name)
                print('value of x vector (R^N) = ', vector)
                with open(filename, 'ab') as f:
                    pickle.dump(vector, f)
                i+=1
    except KeyboardInterrupt:
        print('interrupted !!')
    
    end_time = time.time()
    print("the number of protein processed in current run = ",i)
    print('time elapsed =',end_time-start_time,'seconds')