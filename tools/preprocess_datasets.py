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
import os

import psutil
import ray

from solvent.data.mmcif_parsing import parse
from solvent.data.protein_utils import make_mmcif_features


@ray.remote
def parse_file(
    f,
    args,
    chain_cluster_size_dict
):
    pdb_path = os.path.join(args.pdb_dir, f)
    file_id, ext = os.path.splitext(f)

    if args.fasta_dir is not None:
        fasta_paths = glob.glob(os.path.join(args.fasta_dir, file_id+'_'+'*.fasta'))
        fasta_ids = [os.path.basename(os.path.splitext(fasta_path)[0]) for fasta_path in fasta_paths]
    else:
        fasta_ids = None

    with open(pdb_path, "r") as fp:
        mmcif_string = fp.read()
    mmcif = parse(file_id=file_id, mmcif_string=mmcif_string)

    if mmcif.mmcif_object is None:
        logging.info(f"Could not parse {f}. Skipping...")
        return {}
    else:
        mmcif = mmcif.mmcif_object

    out = {}
    for chain_id, seq in mmcif.chain_to_seqres.items():
        full_name = "_".join([file_id, chain_id])
        if (fasta_ids is not None) and (full_name not in fasta_ids):
            continue

        out[full_name] = {}
        local_data = out[full_name]
        local_data["release_date"] = mmcif.header["release_date"]
        local_data["seq"] = seq
        local_data["resolution"] = mmcif.header["resolution"]

        if (chain_cluster_size_dict is not None):
            cluster_size = chain_cluster_size_dict.get(
                full_name.upper(), -1
            )
            local_data["cluster_size"] = cluster_size

        cache_data = copy.deepcopy(local_data)

        data = make_mmcif_features(
            mmcif_object=mmcif,
            chain_id=chain_id,
        )
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
