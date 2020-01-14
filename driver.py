#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:15:20 2020

@author: saint8312
"""

import pdb_processor as pdbp
import feature_calculator as fc
import time
import multiprocessing

if __name__=="__main__":
    import json

    with open('config.json') as json_data_file:
        conf = json.load(json_data_file)
    x_path = conf['root']['PP']
    y_path = conf['index']['PP']    
    
    complex_files = pdbp.list_files(x_path)
    id_name = complex_files[1]
    test_file = x_path+"/"+id_name
    chains = pdbp.loader_pdbbind(test_file)    
    atom_types = ['C','N','O','S']
    cutoff = 12
    
    print(chains, len(chains))
    start_time = time.time()
    pool = multiprocessing.Pool()    
    
    x_vec = fc.x_processor_mp([chains, id_name, atom_types, cutoff, pool])
    print(x_vec)

    end_time = time.time()
    print('time elapsed =',end_time-start_time,'seconds')