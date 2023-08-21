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
from datetime import datetime

from detectron2.data import DatasetCatalog, MetadataCatalog


def load_casp14(data_dir):
    cache_path = os.path.join(data_dir, 'chain_data_cache.json')
    with open(cache_path, 'r') as f:
        data = json.load(f)

    dataset_dicts = []
    for full_name, info in data.items():
        records = {}

        if full_name == 'T1044_A':
            continue

        file_id, chain_id = full_name.split('_')
        records['data_name'] = 'casp14'
        records['full_name'] = full_name
        records['file_id'] = file_id
        records['chain_id'] = chain_id
        records['file_name'] = os.path.join(data_dir, 'parsed_data', full_name+'.json')
        records['data_weight'] = 1.0

        dataset_dicts.append(records)

    return dataset_dicts


def register_casp14(name, data_dir):
    DatasetCatalog.register(name, lambda: load_casp14(data_dir))
    MetadataCatalog.get(name).set(
        evaluator_type="protein_folding",
    )

dataset_name = 'casp14'
dataset_dir = 'casp14'
register_casp14(dataset_name, os.path.join('datasets', dataset_dir))