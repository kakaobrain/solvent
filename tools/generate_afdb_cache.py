import argparse
import glob
import gzip
import json
import os
import time

import numpy as np
import pandas as pd
from biotite.structure import get_chains
from biotite.structure.io import pdbx


def generate_data_cache(uniprot_id, frag_num, cif_dir):
    
    cif_data_path = glob.glob(os.path.join(cif_dir, '*', f"AF-{uniprot_id}-F{frag_num}-model_v3.cif.gz"))
    confidence_path = glob.glob(os.path.join(cif_dir, '*', f"AF-{uniprot_id}-F{frag_num}-confidence_v3.json.gz"))
    assert len(cif_data_path) == 1
    assert len(confidence_path) == 1
    cif_data_path = cif_data_path[0]
    confidence_path = confidence_path[0]

    cache_dict = {}
    
    with gzip.open(confidence_path, 'rb') as handle:
        confidence = json.load(handle)
        confidence_score = confidence['confidenceScore']
    
    mean_score = np.mean(confidence_score)
    
    with gzip.open(cif_data_path, 'rt') as handle:
        ciffile = pdbx.PDBxFile.read(handle)
        structure = pdbx.get_structure(ciffile, model=1)
    
        all_chains = get_chains(structure)
        num_chains = len(all_chains)
        
        if num_chains==0:
            raise ValueError("No chain is found in the input file.")
        else:
            for chain in all_chains:
                line = {
                    'uniprot_id': uniprot_id, 
                    'frag_num': 1, 
                    'chain_id': chain,
                    'mean_plddt': mean_score
                }
                
                full_name = uniprot_id + '_' + chain
                cache_dict[full_name] = line
            
    return cache_dict

def extract_info_uniref50(args):
    
    uniref_list_path = os.path.join(args.data_dir, 'uniref_lists.txt')
    with open(uniref_list_path, 'r') as f:
        uniprot_ids = f.read()
    uniprot_ids = uniprot_ids.strip().split(' ')

    cif_dir = os.path.join(args.data_dir, 'gz')
    cache_data = [generate_data_cache(uniprot_id, 1, cif_dir) for uniprot_id in uniprot_ids]
    
    output_dir = args.json_data_dir
    with open(os.path.join(output_dir, "chain_data_cache.json"), 'w') as fw:
        fw.write(json.dumps(cache_data, indent=4))

if __name__ == '__main__':
    p = argparse.ArgumentParser()

    p.add_argument('--data_dir', type=str, default='datasets/afdb', help='dataset name')
    p.add_argument('--json_data_dir', type=str, default='datasets/afdb', help='directory containing json files')

    args = p.parse_args()

    start_time = time.time()
    extract_info_uniref50(args)
    print(f"It took {time.time()-start_time:.2f} seconds!")