# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 18:13:15 2020

@author: Saint8312

"""

"""
dedicated file which contain functions to process custom PDB files
req : biopandas
"""

from biopandas.pdb import PandasPdb
import os


def list_files(root_path):
    return [f for f in os.listdir(root_path) if os.path.isfile(os.path.join(root_path, f))]

def load_custom_pdb(filepath):
    '''
    get the 'ATOM' key and set the line index as default index
    '''
    ppdb = PandasPdb()
    ppdb.read_pdb(test_file)
    df_atoms = ppdb.df["ATOM"].set_index(["line_idx"])
    return df_atoms
    
def get_chain_terminals(filepath):
    """
    get the terminal indexes in the case of complexes with multiple protein chains
    """
    idxes = []
    idx = 0
    with open(test_file, 'r') as f:
        for line in f:
            if line.startswith('TER'):
                idxes.append(idx)
            idx+=1
    return idxes
    
def get_sliced_chains(df_atoms, terminal_idxes, zdock=False):
    '''
    slice the chains into list based on the "TER" indexes.
    in case of zdock, atom types need to be added manually
    '''
    chains = []
    split_idxes = []
    if zdock:
        #add the atom types to 'element_symbol'
        df_atoms["element_symbol"] = df_atoms["atom_name"].apply(lambda x:x[0])
        #get the terminal indexes
        atom_numbers = df_atoms["atom_number"]
        length = atom_numbers.shape[0]
        for i in range(length-1):
            if atom_numbers[i+1]<atom_numbers[i]:
                split_idxes.append(i)
        split_idxes.append(length)
    else:
        split_idxes = terminal_idxes
    
    start_idx = 0
    for idx in split_idxes:
        chain=df_atoms.loc[start_idx:idx]
        chains.append(chain)
        start_idx = idx+1
    return chains


#####################################
'''
specific case/test functions
'''
def loader_pdbbind(filepath):
    df_atoms = load_custom_pdb(filepath)
    terminal_idxes = get_chain_terminals(filepath)
    return get_sliced_chains(df_atoms, terminal_idxes)
    
def loader_zdock(filepath):
    df_atoms = load_custom_pdb(filepath)
    terminal_idxes = get_chain_terminals(filepath)
    return get_sliced_chains(df_atoms, terminal_idxes, zdock=True)


if __name__ == '__main__':
#    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Dataset/PP'
    path = 'C:/Users/beryl/Documents/Computational Science/Kanazawa/Thesis/Outputs/PC-PC/top_preds'
    
    complex_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    
#    test_file = path+'/'+complex_files[2]
#    test_file = path+'/1nez.ent.pdb'
    test_file = path+'/complex.1.pdb'
    
    df_atoms = load_custom_pdb(test_file)
    terminal_idxes = get_chain_terminals(test_file)
    chains = get_sliced_chains(df_atoms, terminal_idxes, zdock=True)
    
#    new_chain = chains[0].set_index(["residue_name"])
    chains = loader_zdock(test_file)
    print(chains)
    