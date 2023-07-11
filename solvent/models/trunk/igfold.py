# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from IgFold (https://github.com/Graylab/IgFold)
# Copyright 2022 The Johns Hopkins University
# https://github.com/Graylab/IgFold/blob/main/LICENSE.md

import sys
from functools import partial
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange, repeat

from igfold.model.components import IPAEncoder, TriangleGraphTransformer
from igfold.training.utils import kabsch
from igfold.utils.coordinates import get_ideal_coords
from solvent.models.primitives import LayerNorm, Linear

from . import TRUNK, TRUNK_REGISTRY


def get_coords_tran_rot(
        temp_coords,
        batch_size,
        seq_len,
        device,
        center=True,
    ):
        res_coords = temp_coords
        res_ideal_coords = repeat(
            get_ideal_coords(center=center),
            "a d -> b l a d",
            b=batch_size,
            l=seq_len,
        ).to(device)
        _, rotations, translations = kabsch(
            res_ideal_coords,
            res_coords,
            return_translation_rotation=True,
        )
        translations = rearrange(
            translations,
            "b l () d -> b l d",
        )

        return translations, rotations


@TRUNK_REGISTRY.register()
class IGFoldTrunk(TRUNK):
    def __init__(self, cfg):
        super().__init__()

        num_blocks = cfg.MODEL.TRUNK.NUM_BLOCKS
        node_dim = cfg.MODEL.TRUNK.MSA_DIM
        edge_dim = cfg.MODEL.TRUNK.PAIR_DIM
        single_dim = cfg.MODEL.TRUNK.SINGLE_DIM
        gt_depth = 1
        gt_heads = cfg.MODEL.EMBEDDER.NUM_HEADS
        gt_dim_head = edge_dim // 2
        c_hidden_mul = cfg.MODEL.TRUNK.TRIANGLE.DIM
        no_qk_points = cfg.MODEL.FOLDING.IPA.NUM_SCALAR_QK
        no_v_points = cfg.MODEL.FOLDING.IPA.NUM_POINT_V
        temp_num_blocks = cfg.MODEL.TRUNK.TEMPLATE_IPA.NUM_BLOCKS
        ipa_heads = cfg.MODEL.TRUNK.TEMPLATE_IPA.NUM_HEADS

        self.str_node_transform = nn.Sequential(
            Linear(node_dim, node_dim),
            nn.ReLU(),
            LayerNorm(node_dim),
        )
        self.str_edge_transform = nn.Sequential(
            Linear(edge_dim, edge_dim),
            nn.ReLU(),
            LayerNorm(edge_dim),
        )

        self.main_block = TriangleGraphTransformer(
            dim=node_dim,
            edge_dim=edge_dim,
            depth=num_blocks,
            tri_dim_hidden=2 * node_dim,
            gt_depth=gt_depth,
            gt_heads=gt_heads,
            gt_dim_head=node_dim // 2,
        )

        self.template_ipa = IPAEncoder(
            dim=node_dim,
            pairwise_repr_dim=edge_dim,
            depth=temp_num_blocks,
            heads=ipa_heads,
            require_pairwise_repr=True,
        )

        self.linear = Linear(node_dim, single_dim)

    def forward(self, outputs, feats, prevs, inplace_safe=False, offload_inference=False):
        nodes = feats['msa'].squeeze(1)
        edges = feats['pair']
        mask = feats["seq_mask"].bool()

        
        temp_coords = feats["template_all_atom_positions"][..., :4, :]
        temp_mask = feats["template_all_atom_mask"][..., :4]
        res_temp_mask = temp_mask.all(-1)
        batch_size, seq_len, dim, _ = temp_coords.shape
        temp_translations, temp_rotations = get_coords_tran_rot(
            temp_coords,
            batch_size,
            seq_len,
            device=temp_coords.device
        )

        nodes = self.str_node_transform(nodes)
        edges = self.str_edge_transform(edges)
        nodes, edges = self.main_block(
            nodes,
            edges,
            mask=mask,
        )

        nodes = self.template_ipa(
            nodes,
            translations=temp_translations,
            rotations=temp_rotations,
            pairwise_repr=edges,
            mask=res_temp_mask,
        )
        single = self.linear(nodes)

        outputs['msa'] = nodes.unsqueeze(1)
        outputs['pair'] = edges
        outputs['single'] = single
        return outputs