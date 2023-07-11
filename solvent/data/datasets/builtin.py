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

import os

from .cameo import register_cameo
from .pdb import register_pdb
from .sabdab import register_sabdab
from .uniref50 import register_uniref50

ROOT = 'datasets'

_PREDEFINED_SPLITS_PROTEIN = {}
_PREDEFINED_SPLITS_PROTEIN["pdb"] = {
    "pdb": "pdb_mmcif",
}
_PREDEFINED_SPLITS_PROTEIN["cameo"] = {
    "cameo": "cameo",
}
_PREDEFINED_SPLITS_PROTEIN["uniref50"] = {
    "af2_uniref50": "afdb",
}
_PREDEFINED_SPLITS_PROTEIN["sabdab"] = {
    "sabdab_20210331": "SAbDab/20210331",
    "sabdab_igfold": "SAbDab/IgFold-Benchmark",
}


def register_all_protein(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_PROTEIN.items():
        for key, pdb_file in splits_per_dataset.items():
            if 'pdb' in key:
                register_pdb(
                    key, 
                    os.path.join(root, pdb_file)
                )
            elif 'cameo' in key:
                register_cameo(
                    key,
                    os.path.join(root, pdb_file)
                )
            elif 'uniref50' in key:
                register_uniref50(
                    key, 
                    os.path.join(root, pdb_file)
                )
            elif 'sabdab' in key:
                register_sabdab(
                    key, 
                    os.path.join(root, pdb_file)
                )

register_all_protein(ROOT)
