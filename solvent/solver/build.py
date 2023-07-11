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

import copy
import itertools
import logging
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Type, Union

import torch
from detectron2.config import CfgNode
from detectron2.solver import get_default_optimizer_params
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.solver.lr_scheduler import (
    LRMultiplier,
    LRScheduler,
    WarmupParamScheduler
)
from fvcore.common.param_scheduler import (
    CosineParamScheduler,
    MultiStepParamScheduler,
    StepWithFixedGammaParamScheduler
)


def build_optimizer(cfg: CfgNode, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.
    """
    params = get_default_optimizer_params(
        model,
        base_lr=cfg.SOLVER.BASE_LR,
        weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
        bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
        weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    )
    if cfg.SOLVER.OPTIM == 'Adam':
        optimizer = torch.optim.Adam
    elif cfg.SOLVER.OPTIM == 'RAdam':
        optimizer = torch.optim.Adam

    return maybe_add_gradient_clipping(cfg, optimizer)(
        params,
        lr=cfg.SOLVER.BASE_LR,
        eps=1e-5
    )

def build_lr_scheduler(cfg: CfgNode, optimizer: torch.optim.Optimizer) -> LRScheduler:
    """
    Build a LR scheduler from config.
    """
    name = cfg.SOLVER.LR_SCHEDULER_NAME

    if name == "WarmupMultiStepLR":
        steps = [x for x in cfg.SOLVER.STEPS if x <= cfg.SOLVER.MAX_ITER]
        if len(steps) != len(cfg.SOLVER.STEPS):
            logger = logging.getLogger(__name__)
            logger.warning(
                "SOLVER.STEPS contains values larger than SOLVER.MAX_ITER. "
                "These values will be ignored."
            )
        sched = MultiStepParamScheduler(
            values=[cfg.SOLVER.GAMMA**k for k in range(len(steps) + 1)],
            milestones=steps,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    elif name == "WarmupCosineLR":
        end_value = cfg.SOLVER.BASE_LR_END / cfg.SOLVER.BASE_LR
        assert end_value >= 0.0 and end_value <= 1.0, end_value
        sched = CosineParamScheduler(1, end_value)
    elif name == "WarmupStepWithFixedGammaLR":
        sched = StepWithFixedGammaParamScheduler(
            base_value=1.0,
            gamma=cfg.SOLVER.GAMMA,
            num_decays=cfg.SOLVER.NUM_DECAYS,
            num_updates=cfg.SOLVER.MAX_ITER,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))

    sched = WarmupParamScheduler(
        sched,
        cfg.SOLVER.WARMUP_FACTOR,
        min(cfg.SOLVER.WARMUP_ITERS / cfg.SOLVER.MAX_ITER, 1.0),
        cfg.SOLVER.WARMUP_METHOD,
        cfg.SOLVER.RESCALE_INTERVAL,
    )
    return LRMultiplier(optimizer, multiplier=sched, max_iter=cfg.SOLVER.MAX_ITER)
