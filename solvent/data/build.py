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

import itertools
import logging
from functools import partial

import torch
import torch.utils.data as torchdata
from detectron2.data import DatasetCatalog, MapDataset, ToIterableDataset
from detectron2.utils.comm import get_world_size
from detectron2.utils.env import seed_all_rng

from solvent.data import DatasetFromList
from solvent.utils.tensor_utils import dict_multimap, tensor_tree_map

from .dataset_mapper import DatasetMapper
from .samplers import (
    InferenceSampler,
    TrainingSampler,
    WeightedTrainingSampler
)


def get_folding_dataset_dicts(names, use_weight=False, check_consistency=True):
    """
    Load and prepare dataset dicts.

    Args:
        names (str or list[str]): a dataset name or a list of dataset names
        check_consistency (bool): whether to check if datasets have consistent metadata.

    Returns:
        list[dict]: a list of dicts following the standard dataset dict format.
    """
    if isinstance(names, str):
        names = [names]
    assert len(names), names
    dataset_dicts = [DatasetCatalog.get(dataset_name) for dataset_name in names]

    total_num_per_dataset = [len(d) for d in dataset_dicts]
    total_num = sum(total_num_per_dataset)

    if use_weight:
        weights = []
        for d in dataset_dicts:
            d_num = len(d)
            weight = [total_num * d[0]['data_weight'] / d_num] * d_num
            weights.extend(weight)

    if isinstance(dataset_dicts[0], torchdata.Dataset):
        if len(dataset_dicts) > 1:
            # ConcatDataset does not work for iterable style dataset.
            # We could support concat for iterable as well, but it's often
            # not a good idea to concat iterables anyway.
            return torchdata.ConcatDataset(dataset_dicts)
        return dataset_dicts[0]

    for dataset_name, dicts in zip(names, dataset_dicts):
        assert len(dicts), "Dataset '{}' is empty!".format(dataset_name)

    dataset_dicts = list(itertools.chain.from_iterable(dataset_dicts))

    assert len(dataset_dicts), "No valid data found in {}.".format(",".join(names))
    
    if use_weight:
        return dataset_dicts, weights
    else:
        return dataset_dicts


def build_batch_data_loader(
    dataset,
    sampler,
    total_batch_size,
    *,
    num_workers=0,
    collate_fn=None,
):
    """
    Build a batched dataloader. The main differences from `torch.utils.data.DataLoader` are:
    1. support aspect ratio grouping options
    2. use no "batch collation", because this is common for detection training

    Args:
        dataset (torch.utils.data.Dataset): a pytorch map-style or iterable dataset.
        sampler (torch.utils.data.sampler.Sampler or None): a sampler that produces indices.
            Must be provided iff. ``dataset`` is a map-style dataset.
        total_batch_size, aspect_ratio_grouping, num_workers, collate_fn: see
            :func:`build_detection_train_loader`.

    Returns:
        iterable[list]. Length of each list is the batch size of the current
            GPU. Each element in the list comes from the dataset.
    """
    world_size = get_world_size()
    assert (
        total_batch_size > 0 and total_batch_size % world_size == 0
    ), "Total batch size ({}) must be divisible by the number of gpus ({}).".format(
        total_batch_size, world_size
    )
    batch_size = total_batch_size // world_size

    if isinstance(dataset, torchdata.IterableDataset):
        assert sampler is None, "sampler must be None if dataset is IterableDataset"
    else:
        dataset = ToIterableDataset(dataset, sampler)

    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
        worker_init_fn=worker_init_reset_seed,
        pin_memory=True,
    )


def build_folding_train_loader(cfg, mapper=None):
    dataset, weights = get_folding_dataset_dicts(cfg.DATASETS.TRAIN, use_weight=True)
    dataset = DatasetFromList(dataset)

    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=True)
    dataset = MapDataset(dataset, mapper)
    
    sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
    logger = logging.getLogger(__name__)
    if isinstance(dataset, torchdata.IterableDataset):
        logger.info("Not using any sampler since the dataset is IterableDataset.")
        sampler = None
    else:
        logger.info("Using training sampler {}".format(sampler_name))
        if sampler_name == "TrainingSampler":
            sampler = TrainingSampler(len(dataset))
        elif sampler_name == "WeightedTrainingSampler":
            sampler = WeightedTrainingSampler(weights, len(dataset))
        else:
            raise ValueError("Unknown training sampler: {}".format(sampler_name))
    
    total_batch_size = cfg.SOLVER.SEQ_PER_BATCH
    num_workers = cfg.DATALOADER.NUM_WORKERS

    return build_batch_data_loader(
        dataset,
        sampler,
        total_batch_size,
        num_workers=num_workers,
        collate_fn=sample_iter_batch_collator
    )

def build_folding_test_loader(cfg, mapper=None):
    dataset = get_folding_dataset_dicts(cfg.DATASETS.TEST)
    dataset = DatasetFromList(dataset, copy=False)

    if mapper is None:
        mapper = DatasetMapper(cfg, is_train=False)
    dataset = MapDataset(dataset, mapper)

    sampler = InferenceSampler(len(dataset))

    batch_size = 1
    num_workers = cfg.DATALOADER.NUM_WORKERS
    
    return torchdata.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=None,
        pin_memory=True,
    )


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    stack_fn = partial(torch.stack, dim=0)
    return dict_multimap(stack_fn, batch)

def sample_iter_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    
    stack_fn = partial(torch.stack, dim=0)
    batch = dict_multimap(stack_fn, batch)

    max_iters = batch['aatype'].shape[-1]
    recycling_probs = [
                1. / (max_iters + 1) for _ in range(max_iters + 1)
            ]
    num_iter = torch.multinomial(torch.tensor(recycling_probs), num_samples=1, replacement=True) + 1
    
    batch = tensor_tree_map(lambda x: x[..., :num_iter], batch)
    return batch

def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)