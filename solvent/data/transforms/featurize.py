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

from typing import Dict, Mapping

import ml_collections
import numpy as np
import torch

from . import FEAT, NUM_EXTRA_SEQ, NUM_MSA_SEQ, NUM_RES, NUM_TEMPLATES
from . import transform as data_transforms

FeatureDict = Mapping[str, np.ndarray]
TensorDict = Dict[str, torch.Tensor]



def nonensembled_transform_fns(config):
    """Input pipeline data transformers that are not ensembled."""
    transforms = [
        data_transforms.cast_to_64bit_ints,
        data_transforms.correct_msa_restypes,
        data_transforms.squeeze_features,
        data_transforms.randomly_replace_msa_with_unknown(0.0),
        data_transforms.make_seq_mask,
        data_transforms.make_msa_mask,
        data_transforms.make_hhblits_profile,
    ]
    if config['use_template']:
        transforms.extend(
            [
                data_transforms.fix_templates_aatype,
                data_transforms.make_template_mask,
                data_transforms.make_pseudo_beta("template_"),
            ]
        )
        if config['use_template_torsion_angles']:
            transforms.extend(
                [
                    data_transforms.atom37_to_torsion_angles("template_"),
                ]
            )

    transforms.extend(
        [
            data_transforms.make_atom14_masks,
        ]
    )
    if config['is_supervised']:
        transforms.extend(
            [
                data_transforms.make_atom14_positions,
                data_transforms.atom37_to_frames,
                data_transforms.atom37_to_torsion_angles(""),
                data_transforms.make_pseudo_beta(""),
                data_transforms.get_backbone_frames,
                data_transforms.get_chi_angles,
            ]
        )

    return transforms


def ensembled_transform_fns(config, ensemble_seed):
    """Input pipeline data transformers that can be ensembled and averaged."""
    transforms = []

    if "max_distillation_msa_clusters" in config:
        transforms.append(
            data_transforms.sample_msa_distillation(
                config.max_distillation_msa_clusters
            )
        )

    max_msa_clusters = pad_msa_clusters = config['max_msa_clusters']
    max_extra_msa = False

    if config['masked_msa']:
        # Masked MSA should come *before* MSA clustering so that
        # the clustering and full MSA profile do not leak information about
        # the masked locations and secret corrupted locations.
        transforms.append(
            data_transforms.make_masked_msa(
                config
            )
        )

    if 'msa_cluster_features' in config:
        transforms.append(data_transforms.nearest_neighbor_clusters())
        transforms.append(data_transforms.summarize_clusters())

    # Crop after creating the cluster profiles.
    if max_extra_msa:
        transforms.append(data_transforms.crop_extra_msa(max_extra_msa))
    else:
        transforms.append(data_transforms.delete_extra_msa)

    transforms.append(data_transforms.make_msa_feat())


    if config['is_train']:
        transforms.append(data_transforms.select_feat(list(FEAT)))
        transforms.append(
            data_transforms.random_crop_to_size(
                config['crop_size'],
                config['max_templates'],
                FEAT,
                seed=ensemble_seed + 1,
            )
        )
        transforms.append(
            data_transforms.make_fixed_size(
                FEAT,
                pad_msa_clusters,
                config['max_extra_msa'],
                config['crop_size'],
                config['max_templates'],
            )
        )
    else:
        # make fixed size input is necessary even when evaluation because of xformers.
        # padding should be removed when evaluation
        transforms.append(data_transforms.select_feat(list(FEAT)))
        transforms.append(
            data_transforms.make_fixed_size(
                FEAT,
                pad_msa_clusters,
                config['max_extra_msa'],
                config['crop_size'],
                config['max_templates'],
            )
        )
        transforms.append(
            data_transforms.crop_templates(config['max_templates'])
        )

    return transforms

def process_tensors_from_config(tensors, config):
    """Based on the config, apply filters and transformations to the data."""

    ensemble_seed = torch.Generator().seed()

    def wrap_ensemble_fn(data, i):
        """Function to be mapped over the ensemble dimension."""
        d = data.copy()
        fns = ensembled_transform_fns(
            config,
            ensemble_seed,
        )
        fn = compose(fns)
        d["ensemble_index"] = i
        return fn(d)

    no_templates = True
    if("template_aatype" in tensors):
        no_templates = tensors["template_aatype"].shape[0] == 0

    nonensembled = nonensembled_transform_fns(config)

    tensors = compose(nonensembled)(tensors)

    if("no_recycling_iters" in tensors):
        num_recycling = int(tensors["no_recycling_iters"])
    else:
        num_recycling = config['max_recycling_iters']

    tensors = map_fn(
        lambda x: wrap_ensemble_fn(tensors, x), torch.arange(num_recycling + 1)
    )

    return tensors

@data_transforms.curry1
def compose(x, fs):
    for f in fs:
        x = f(x)
    return x


def map_fn(fun, x):
    ensembles = [fun(elem) for elem in x]
    features = ensembles[0].keys()
    ensembled_dict = {}
    for feat in features:
        ensembled_dict[feat] = torch.stack(
            [dict_i[feat] for dict_i in ensembles], dim=-1
        )
    return ensembled_dict
