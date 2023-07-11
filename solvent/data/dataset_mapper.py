# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from Detectron2 (https://github.com/facebookresearch/detectron2)
# Copyright (c) Facebook, Inc. and its affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gzip
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import torch
from Bio.Data import SCOPData
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
from detectron2.config import configurable

from solvent.common import residue_constants
from .protein_utils import (
    make_sequence_features,
    process_msa_feats
)
from .transforms.featurize import process_tensors_from_config


class DatasetMapper:
    @configurable
    def __init__(self, config, is_train: bool):
        self.is_train = is_train
        self.config = config
    
    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = {
            "is_train": is_train,
        }
        
        feature_names = cfg.INPUT.UNSUPERVISED_FEATURES    
        # if is_train:
        feature_names += cfg.INPUT.SUPERVISED_FEATURES

        config = {
            'is_supervised': True,
            'is_train': is_train,
            'feature_names': feature_names,
            'use_template': cfg.INPUT.USE_TEMPLATE,
            'max_recycling_iters': cfg.MODEL.NUM_RECYCLE,
            'max_msa_clusters': cfg.INPUT.MAX_MSA_CLUSTERS,
            'max_extra_msa': cfg.INPUT.MAX_EXTRA_MSA,
            'clamp_prob': cfg.SOLVER.CLAMP_PROB
        }
        if cfg.MODEL.HEAD.MASKED_MSA.ENABLED:
            config['masked_msa'] = True
            config['uniform_prob'] = cfg.MODEL.HEAD.MASKED_MSA.UNIFORM_PROB
            config['profile_prob'] = cfg.MODEL.HEAD.MASKED_MSA.PROFILE_PROB
            config['same_prob'] = cfg.MODEL.HEAD.MASKED_MSA.SAME_PROB
            config['replace_fraction'] = cfg.MODEL.HEAD.MASKED_MSA.REPLACE_FRACTION
        
        if cfg.INPUT.CROP.ENABLED:
            config['crop_enabled'] = True
            crop_size = cfg.INPUT.CROP.SIZE
            config['crop_size'] = crop_size
            config['max_templates'] = cfg.INPUT.CROP.MAX_TEMPLATE
        else:
            config['crop_enabled'] = False
            config['max_templates'] = cfg.INPUT.CROP.MAX_TEMPLATE

        ret['config'] = config

        return ret

    def _parse_data(self, dataset_dict):
        data_name = dataset_dict['data_name']
        if data_name == 'alphafold':
            uniprot_id = dataset_dict['file_id']
            chain_id = dataset_dict['chain_id']
            frag_num = dataset_dict['frag_num']
            full_name = dataset_dict['full_name']

            data_path = dataset_dict['file_name']
            with gzip.open(data_path, 'rt') as handle:
                ciffile = pdbx.PDBxFile.read(handle)
                structure = pdbx.get_structure(ciffile, model=1)

                chain_filter = [a.chain_id in chain_id for a in structure]

                structure = structure[chain_filter]
                residue_identities = get_residues(structure)[1]
                seq = ''.join([ProteinSequence.convert_letter_3to1(r) for r in residue_identities])

                # mmcif feats
                num_res = len(seq)

                data = make_sequence_features(seq, full_name, num_res)
                all_atom_positions, all_atom_mask = get_atom_coords(seq, structure)
                data["all_atom_positions"] = all_atom_positions
                data["all_atom_mask"] = all_atom_mask
                data['resolution'] = np.array([1.], dtype=np.float32)
                data['is_distillation'] = np.array(1., dtype=np.float32)
                data.update(process_msa_feats(seq))

        else:
            file_id = dataset_dict['file_id']
            chain_id = dataset_dict['chain_id']
            full_name = dataset_dict['full_name']

            path = dataset_dict['file_name']
            with open(path, 'r') as f:
                data = json.load(f)[full_name]
            seq = data['seq']
            data.update(process_msa_feats(seq))

        return data

    def __call__(self, dataset_dict):
        
        np_example = self._parse_data(dataset_dict)
        
        num_res = int(np_example["seq_length"][0])
        if not self.config['crop_enabled']:
            self.config['crop_size'] = num_res

        if not self.config['is_train']:
            self.config['crop_size'] = (int(num_res / 8) + 1) * 8

        if "deletion_matrix_int" in np_example:
            np_example["deletion_matrix"] = np_example.pop(
                "deletion_matrix_int"
            ).astype(np.float32)
        
        # numpy to tensor dict
        features = {
            k: torch.tensor(v) for k, v in np_example.items() if k in self.config['feature_names']
        }

        # feature is cropped as config['crop_size'] only when training.
        # when eval or test, feature size is kept as 'num_res'.
        features = process_tensors_from_config(features, self.config)

        if self.is_train:
            p = torch.rand(1).item()
            use_clamped_fape_value = float(p < self.config['clamp_prob'])
            features["use_clamped_fape"] = torch.full(
                size=[self.config['max_recycling_iters'] + 1],
                fill_value=use_clamped_fape_value,
                dtype=torch.float32,
            )
        else:
            features["use_clamped_fape"] = torch.full(
                size=[self.config['max_recycling_iters'] + 1],
                fill_value=0.0,
                dtype=torch.float32,
            )
        features = {k: v for k, v in features.items()}

        return features

def get_atom_coords(input_sequence, structure):
    num_res = len(input_sequence)

    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )

    for res_index in range(num_res):
        restype = input_sequence[res_index]
        resname = residue_constants.restype_1to3[restype]
        
        res_index_mask = structure.res_id == (res_index + 1)
        structure_res = structure[res_index_mask]

        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
       
        for atom_name in residue_constants.residue_atoms[resname]:
            atom_mask = structure_res.atom_name == atom_name
            struct_atom = structure_res[atom_mask]
            x, y, z = struct_atom.coord[0]
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name.upper() == "SE" and res.get_resname() == "MSE":
                # Put the coords of the selenium atom in the sulphur column
                pos[residue_constants.atom_order["SD"]] = [x, y, z]
                mask[residue_constants.atom_order["SD"]] = 1.0

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    return all_atom_positions, all_atom_mask