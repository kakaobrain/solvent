# Copyright (c) 2023 Kakao Brain. All Rights Reserved.

import argparse
import os
from datetime import datetime
import glob

import pandas as pd
from tqdm import tqdm


def parse_sabdab_summary(summary_file_path):
    sabdab_summary = {}
    
    df = pd.read_csv(summary_file_path, delimiter='\t', keep_default_na=False)
    
    for index, row in df.iterrows():
        pdb_id = row['pdb']
        if pdb_id not in sabdab_summary:
            sabdab_summary[pdb_id] = []
        
        info = {}
        for col in df.columns:
            info[col] = row[col]
        sabdab_summary[pdb_id].append(info)
    
    return sabdab_summary

def truncate_chain(pdb_string, chain, limit, chain_id):
    truncated_string = ""

    prev = -1
    pdb_string_cmp = pdb_string.split('\n')
    for line in pdb_string_cmp:
        is_atom = line.startswith("ATOM")
        if not is_atom:
            continue

        residx = int(line[22:26])
        is_target_chain = line[21] == chain
        is_fv = residx <= limit
        if (is_atom and is_target_chain and is_fv):

            cur = residx
            if prev > cur:
                # Same chain pairs exist in the one pdb id. 
                # we use only first pairs
                break

            truncated_string += line[:21] + chain_id + line[22:]
            truncated_string += "\n"

            prev = cur

    return truncated_string

def truncate_fv(pdb_id, db_path, output_path, sabdab_summary, end_date):

    pdb_path = os.path.join(db_path, pdb_id + '.pdb')
    with open(pdb_path, 'r') as f:
        pdb_string = f.read()

    current_year = int(datetime.now().strftime("%y"))
    for pdb_info in sabdab_summary[pdb_id][:1]:     # we use only one sample for one pdb_id

        release_date = pdb_info['date']
        month, day, year = map(int, release_date.split('/'))
        if year <= current_year:
            year += 2000
        else:
            year += 1900
        rdate = datetime(year, month, day)
        end_year, end_month, end_day = map(int, end_date.split('/'))
        edate = datetime(end_year, end_month, end_day)
        if rdate > edate:
            return

        h_chain = pdb_info["Hchain"]
        l_chain = pdb_info["Lchain"]

        # L-chain only nanobody is not used in this project.
        if h_chain == 'NA' and l_chain is not 'NA':
            return

        if h_chain == l_chain:
            return

        ab_chain = ''
        h_chain_string = ''
        l_chain_string = ''
        if not h_chain == 'NA':
            ab_chain += h_chain
            h_chain_string = truncate_chain(pdb_string, h_chain, 112, 'H')
            if len(h_chain_string) == 0:
                return

        if not l_chain == 'NA':
            ab_chain += l_chain
            l_chain_string = truncate_chain(pdb_string, l_chain, 109, 'L')
            if len(l_chain_string) == 0:
                return
        
        fv_string = h_chain_string + l_chain_string

        trunc_pdb_path = os.path.join(os.path.join(output_path, pdb_id + '_' + ab_chain + '.pdb'))
        with open(trunc_pdb_path, 'w') as f:
            f.write(fv_string)


def extract_antibody_fv(db_path, summary_file_path, end_date=None):
    all_pdb_paths = glob.glob(os.path.join(db_path, '*'))
    all_pdb_names = [os.path.splitext(os.path.basename(p))[0] for p in all_pdb_paths]
    all_pdb_names = list(set(all_pdb_names))
    
    output_path = os.path.join(db_path, '..', '..', ''.join(end_date.split('/')), 'pdbs')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    sabdab_summary = parse_sabdab_summary(summary_file_path)

    for pdb_id in tqdm(all_pdb_names):
        truncate_fv(
            pdb_id, 
            db_path, output_path,
            sabdab_summary, 
            end_date, 
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--summary_file',
        default=None,
        help='Summary file to use (will download if not provided)')
    parser.add_argument(
        '--db_path',
        default=None,
        help='db path')
    parser.add_argument(
        '--end_date',
        default=None,
        help='cutoff date')
    args = parser.parse_args()

    extract_antibody_fv(
        db_path=args.db_path,
        summary_file_path=args.summary_file,
        end_date=args.end_date,
    )