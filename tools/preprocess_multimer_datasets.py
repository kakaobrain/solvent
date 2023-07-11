# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from OpenFold (https://github.com/aqlaboratory/openfold)
# Copyright 2021 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import glob
import json
import logging
import numpy as np
import os

import psutil
import ray

from solvent.data.mmcif_parsing import parse
from solvent.data.protein_utils import process_pdb
from igfold.utils.pdb import cdr_indices

@ray.remote
def parse_file(
    f,
    args,
    chain_cluster_size_dict
):
    pdb_path = os.path.join(args.pdb_dir, f)
    full_name = os.path.splitext(os.path.basename(pdb_path))[0]
    
    out = {}
    out[full_name] = {}
    local_data = out[full_name]
    data = process_pdb(
        pdb_path=pdb_path,
        is_distillation=False,
    )

    cdr_index = np.zeros_like(data['residue_index'])
    chain_type = list(set(list(data['chain_index'])))
    if len(chain_type) == 2:
        cdr_names = ["h1", "h2", "h3", "l1", "l2", "l3"]
    else:
        if chain_type[0] == 7:
            cdr_names = ["h1", "h2", "h3"]
        else:
            cdr_names = ["l1", "l2", "l3"]
    
    cdr_h_index = cdr_index[data['chain_index'] == 7]
    cdr_l_index = cdr_index[data['chain_index'] == 11]
    for cdr in cdr_names:
        start_idx, end_idx = cdr_indices(pdb_path, cdr, offset_heavy=False)
        if 'h' in cdr:
            cdr_h_index[start_idx:end_idx+1] = 1
        else:
            cdr_l_index[start_idx:end_idx+1] = 1
    cdr_index = np.concatenate([cdr_h_index, cdr_l_index])

    data.update({'cdr_index': cdr_index})
    local_data['seq'] = data['sequence'][0].decode()
    cache_data = copy.deepcopy(local_data)
    for k, v in data.items():
        if v.dtype == 'O':
            continue
        local_data[k] = v.tolist()

    json_path = os.path.join(args.output_dir, full_name+'.json')
    with open(json_path, "w") as fp:
        fp.write(json.dumps(out, indent=4))

    out[full_name] = cache_data
    return out


def main(args):
    json_dir = args.output_dir
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    chain_cluster_size_dict = None
    if (args.cluster_file is not None):
        chain_cluster_size_dict = {}
        with open(args.cluster_file, "r") as fp:
            clusters = [l.strip() for l in fp.readlines()]

        for cluster in clusters:
            chain_ids = cluster.split()
            cluster_len = len(chain_ids)
            for chain_id in chain_ids:
                chain_id = chain_id.upper()
                chain_cluster_size_dict[chain_id] = cluster_len

    accepted_exts = [".cif", ".pdb"]
    files = list(os.listdir(args.pdb_dir))
    files = [f for f in files if os.path.splitext(f)[-1] in accepted_exts]

    parsed_data = [parse_file.remote(f, args, chain_cluster_size_dict) for f in files]
    parsed_data = ray.get(parsed_data)

    cache_data = {}
    for d in parsed_data:
        cache_data.update(d)

    cache_path = os.path.join(args.output_dir, '..', 'chain_data_cache.json')
    with open(cache_path, "w") as fp:
        fp.write(json.dumps(cache_data, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pdb_dir", type=str, help="Directory containing mmCIF or PDB files"
    )
    parser.add_argument(
        "--fasta_dir", type=str, default=None, help="Directory containing mmCIF or PDB files"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Path for parsed .json output"
    )
    parser.add_argument(
        "--cluster_file", type=str, default=None,
        help=(
            "Path to a cluster file (e.g. PDB40), one cluster "
            "({PROT1_ID}_{CHAIN_ID} {PROT2_ID}_{CHAIN_ID} ...) per line. "
            "Chains not in this cluster file will NOT be filtered by cluster "
            "size."
        )
    )

    args = parser.parse_args()
    num_cpus = psutil.cpu_count()
    ray.init(num_cpus=num_cpus)

    main(args)

    ray.shutdown()
