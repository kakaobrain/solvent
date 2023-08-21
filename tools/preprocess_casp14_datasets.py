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
import dataclasses
import io
from typing import Any, Sequence, Mapping, Optional
import re
import string
import numpy as np
from Bio.PDB import PDBParser

from solvent.data.mmcif_parsing import parse
from solvent.common.protein import Protein
from solvent.common import residue_constants
from solvent.data.protein_utils import (
    _aatype_to_str_sequence,
    make_pdb_features,
    process_msa_feats,
)

FeatureDict = Mapping[str, np.ndarray]


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.

    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.

    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.

    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(
            f"Only single model PDBs are supported. Found {len(models)} models."
        )
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if(chain_id is not None and chain.id != chain_id):
            continue
        for res in chain:
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                mask[residue_constants.atom_order[atom.name]] = 1.0
                res_b_factors[
                    residue_constants.atom_order[atom.name]
                ] = atom.bfactor
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append('A')
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if("PARENT" in pdb_str):
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if("PARENT" in l):
                if(not "N/A" in l):
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([
                        chain_id for _ in parent_names
                    ])
                chain_id += 1

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )

def process_pdb(
        pdb_path: str,
        is_distillation: bool = True,
        chain_id: Optional[str] = None,
        _structure_index: Optional[str] = None,
    ) -> FeatureDict:
        """
            Assembles features for a protein in a PDB file.
        """
        with open(pdb_path, 'r') as f:
            pdb_str = f.read()

        protein_object = from_pdb_string(pdb_str, chain_id)
        input_sequence = _aatype_to_str_sequence(protein_object.aatype) 
        description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
        pdb_feats = make_pdb_features(
            protein_object,
            description,
            is_distillation=is_distillation
        )

        msa_features = process_msa_feats(input_sequence)

        return {**pdb_feats, **msa_features}

def parse_file(
    f,
    args
):
    pdb_path = os.path.join(args.pdb_dir, f)
    file_id, ext = os.path.splitext(f)

    with open(pdb_path, "r") as fp:
        mmcif_string = fp.read()

    local_out = {}
    local_data = {}
    data = process_pdb(
        pdb_path=pdb_path,
        is_distillation=False,
    )
    data.pop('deletion_matrix_int')
    data.pop('msa')
    data['seq'] = np.array(data['sequence'].tolist()[0].decode())
    for k, v in data.items():
        if v.dtype == 'O':
            continue
        local_data[k] = v.tolist()

    full_name = file_id + '_' + 'A'
    json_path = os.path.join(args.output_dir, full_name+'.json')

    local_out[full_name] = local_data
    with open(json_path, "w") as fp:
        fp.write(json.dumps(local_out, indent=4))

    global_out = {}
    global_out[full_name] = {'seq': local_data['seq']}
    return global_out


def main(args):
    json_dir = args.output_dir
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    accepted_exts = [".pdb"]
    files = list(os.listdir(args.pdb_dir))
    files = [f for f in files if os.path.splitext(f)[-1] in accepted_exts]
    parsed_data = [parse_file(f, args) for f in files]

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
        "--output_dir", type=str, help="Path for parsed .json output"
    )
    args = parser.parse_args()

    main(args)