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

from detectron2.utils.registry import Registry

from .folding import FOLDING

FOLDING_REGISTRY = Registry("FOLDING")
FOLDING_REGISTRY.__doc__ = """
Registry for folding, which extract structure from feature map
"""


def build_folding(cfg):
    """
    Build a embedder from `cfg.MODEL.FOLDING.NAME`.

    Returns:
        an instance of :class:`FOLDING`
    """

    folding_name = cfg.MODEL.FOLDING.NAME
    folding = FOLDING_REGISTRY.get(folding_name)(cfg)
    assert isinstance(folding, FOLDING)
    return folding