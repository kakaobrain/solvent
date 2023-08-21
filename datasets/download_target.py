#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import re
import requests

from solvent.data import mmcif_parsing

def read_file_names(file_path):
    with open(file_path, 'r') as file:
        file_names = [line.strip() for line in file.readlines()]
    return file_names

def cuttoff_chain(f_list):
    names = []
    chains = []
    for f_name in f_list:
        name, chain = f_name.split('_')
        names.append(name)
        chains.append(chain)
    return names, chains

def main(args):
    data_dir_path = os.path.join(args.output_dir, "natives")
    fasta_dir_path = os.path.join(args.output_dir, "fastas")

    os.makedirs(data_dir_path, exist_ok=True)
    os.makedirs(fasta_dir_path, exist_ok=True)

    file_name_list = read_file_names(args.target_list_file)
    f_name_list, chain_list = cuttoff_chain(file_name_list)

    error_pdb_list, error_chain_list = [],[]
    for pdb_id, chain_id in zip(f_name_list, chain_list):
        try:
            pdb_url = f"https://files.rcsb.org/view/{pdb_id}.cif"
            pdb_file = requests.get(pdb_url).text
            parsed_cif = mmcif_parsing.parse(
                file_id=pdb_id, mmcif_string=pdb_file
            )
            mmcif_object = parsed_cif.mmcif_object
            # print(mmcif_object)
            if(mmcif_object is None):
                raise list(parsed_cif.errors.values())[0]

            seq = mmcif_object.chain_to_seqres[chain_id]
            fasta_file = '\n'.join([
                f">{pdb_id}_{chain_id}",
                seq,
            ])
            fasta_filename = f"{pdb_id}_{chain_id}.fasta"
            with open(os.path.join(fasta_dir_path, fasta_filename), "w") as fp:
                fp.write(fasta_file)

            cif_filename = f"{pdb_id}.cif"
            with open(os.path.join(data_dir_path, cif_filename), "w") as fp:
                fp.write(pdb_file)
        except:
            error_pdb_list.append(pdb_id)
            error_chain_list.append(chain_id)
            pass

    print("DONE")
    print("Error PDB ", error_pdb_list)
    print("Error chain ", error_chain_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_list_file", type=str, default='datasets/denovo_target.txt')
    parser.add_argument("--output_dir", type=str, default='datasets/denovo')
    args = parser.parse_args()
    main(args)
