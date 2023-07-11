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

from .trunk import TRUNK

TRUNK_REGISTRY = Registry("TRUNK")
TRUNK_REGISTRY.__doc__ = """
Registry for trunk, which extract feature map from representation
"""


def build_trunk(cfg):
    """
    Build a embedder from `cfg.MODEL.TRUNK.NAME`.

    Returns:
        an instance of :class:`TRUNK`
    """

    trunk_name = cfg.MODEL.TRUNK.NAME
    trunk = TRUNK_REGISTRY.get(trunk_name)(cfg)
    assert isinstance(trunk, TRUNK)
    return trunk