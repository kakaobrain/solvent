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

import glob
import json
import os

import pandas as pd
from detectron2.data import DatasetCatalog, MetadataCatalog


def load_uniref50(data_dir):
    cache_data_paths = os.path.join(data_dir, 'chain_data_cache.json')
    with open(cache_data_paths, 'r') as f:
        cache_data = json.load(f)
    
    dataset_dicts = []
    for cache_d in cache_data:
        for full_name, info in cache_d.items():
            records = {}
            uniprot_id = info['uniprot_id']
            frag_num = info['frag_num']
            chain_id = info['chain_id']
            mean_plddt = info['mean_plddt']
            
            if mean_plddt < 70:
                continue

            records['data_name'] = 'alphafold'
            records['chain_id'] = chain_id
            records['full_name'] = uniprot_id + '_' + chain_id
            records['file_name'] = glob.glob(os.path.join(data_dir, 'gz', '*', f"AF-{uniprot_id}-F{frag_num}-model_v3.cif.gz"))[0]
            records['file_id'] = uniprot_id
            records['frag_num'] = frag_num
            records['data_weight'] = 3.0
        
            dataset_dicts.append(records)
    
    return dataset_dicts


def register_uniref50(name, data_dir):
    DatasetCatalog.register(name, lambda: load_uniref50(data_dir))
    MetadataCatalog.get(name).set(
        evaluator_type="protein_folding",
    )