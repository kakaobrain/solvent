# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from IgFold (https://github.com/Graylab/IgFold)
# Copyright 2022 The Johns Hopkins University
# https://github.com/Graylab/IgFold/blob/main/LICENSE.md

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn

from igfold.model.components import IPAEncoder
from igfold.training.utils import (
    kabsch_mse, 
    bond_length_l1, 
    bb_prmsd_l1
)

from ..primitives import LayerNorm, Linear
from .build import HEAD_REGISTRY


@HEAD_REGISTRY.register()
class IGFoldHeads(nn.Module):
    def __init__(self, cfg):
        super(IGFoldHeads, self).__init__()

        tm_enabled = cfg.MODEL.HEAD.TM.ENABLED
        embedder_attn_dim = cfg.MODEL.EMBEDDER.NUM_LAYERS * cfg.MODEL.EMBEDDER.NUM_HEADS
        node_dim = cfg.MODEL.TRUNK.MSA_DIM
        edge_dim = cfg.MODEL.TRUNK.PAIR_DIM
        num_blocks = cfg.MODEL.HEAD.IGFOLD.IPA_NUM_BLOCKS
        dev_ipa_heads = cfg.MODEL.HEAD.IGFOLD.IPA_HEAD
        no_qk_points = cfg.MODEL.FOLDING.IPA.NUM_SCALAR_QK
        no_v_points = cfg.MODEL.FOLDING.IPA.NUM_POINT_V
        plm_heads = cfg.MODEL.EMBEDDER.NUM_HEADS
        self.scale_factor = cfg.MODEL.FOLDING.TRANSITION.SCALE_FACTOR

        self.dev_node_transform = nn.Sequential(
            Linear(node_dim, node_dim),
            nn.ReLU(),
            LayerNorm(node_dim),
        )
        self.dev_edge_transform = nn.Sequential(
            Linear(
                edge_dim,
                edge_dim,
            ),
            nn.ReLU(),
            LayerNorm(edge_dim),
        )
        
        self.dev_ipa = IPAEncoder(
            dim=node_dim,
            pairwise_repr_dim=edge_dim,
            depth=num_blocks,
            heads=dev_ipa_heads,
            require_pairwise_repr=True,
        )
        self.dev_linear = Linear(
            node_dim,
            4,
        )

    def forward(self, outputs, batched_input):
        aux_out = {}

        plm_feats = batched_input['msa'].squeeze(1)
        plm_attn = batched_input['pair']
        mask = batched_input['seq_mask']
        
        ipa_translations = outputs['trans']
        ipa_rotations = outputs['rot']
        
        dev_nodes = self.dev_node_transform(plm_feats)
        dev_edges = self.dev_edge_transform(plm_attn)

        dev_out_feats = self.dev_ipa(
            dev_nodes,
            translations=ipa_translations.detach(),
            rotations=ipa_rotations.detach(),
            pairwise_repr=dev_edges,
            mask=mask.bool(),
        )
        dev_pred = F.relu(self.dev_linear(dev_out_feats))
        dev_pred = rearrange(dev_pred, "b l a -> b (l a)", a=4)

        aux_out["dev_pred"] = dev_pred
        
        prmsd = rearrange(
            dev_pred,
            "b (l a) -> b l a",
            a=4,
        )
        res_rmsd = prmsd.square().mean(dim=-1).sqrt()
        aux_out['plddt'] = res_rmsd
        outputs.update(aux_out)
        
        if self.training:
            losses = self.losses(outputs, batched_input)
            return losses
        else:
            return outputs
    
    def losses(self, out, batch):
        pred_coords = out['final_atom_positions']
        bb_coords = rearrange(
            pred_coords[:, :, :3],
            "b l a d -> b (l a) d",
        )
        flat_coords = rearrange(
            pred_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )

        gt_coords = batch['all_atom_positions'][:, :, :5, :]
        gt_coords_mask = batch['all_atom_mask'][:, :, :5].bool()
        coords_label = rearrange(
            gt_coords[:, :, :4],
            "b l a d -> b (l a) d",
        )
        coords_mask = rearrange(
            gt_coords_mask[:, :, :4],
            "b l a -> b (l a)",
        )
        bb_coords_label = rearrange(
            gt_coords[:, :, :3],
            "b l a d -> b (l a) d")
        bb_batch_mask = rearrange(
            gt_coords_mask[:, :, :3],
            "b l a -> b (l a)")

        coords_loss = kabsch_mse(
                flat_coords,
                coords_label,
                align_mask=coords_mask,
                mask=coords_mask,
                clamp=0.0
        )
        bondlen_loss = bond_length_l1(
                bb_coords,
                bb_coords_label,
                bb_batch_mask,
        )
        bondlen_loss = bondlen_loss.unsqueeze(0).mean(1)
        
        if torch.isnan(bondlen_loss):
            bondlen_loss[0] = 1.0
        else:
            bondlen_loss = bondlen_loss * 10
            bondlen_loss = torch.clamp(bondlen_loss, max=1.0)
    
        prmsd_loss = []
        align_mask = torch.zeros_like(coords_mask)
        res_batch_mask = batch['seq_mask']
        seq_lens = batch['seq_length'].unsqueeze(1)
        
        chain_index = batch['chain_index']
        valid_aa = batch['seq_mask']
        valid_chain_idx = torch.logical_and(chain_index, valid_aa)
        chain_ids = chain_index[valid_chain_idx].unique()   # 7, 11

        seq_lens = []
        for chain_id in chain_ids:
            masking = (chain_index == chain_id).int()
            res_len = masking.sum()
            seq_lens.append(res_len.item())

        prmsd_loss = []
        align_mask = coords_mask.clone()
        cum_seq_lens = np.cumsum([0] + seq_lens)
        for sl_i, sl in enumerate(seq_lens):
            align_mask_ = align_mask.clone()
            align_mask_[:, :cum_seq_lens[sl_i]] = False
            align_mask_[:, cum_seq_lens[sl_i + 1]:] = False
            res_batch_mask_ = res_batch_mask.clone()
            res_batch_mask_[:, :cum_seq_lens[sl_i]] = False
            res_batch_mask_[:, cum_seq_lens[sl_i + 1]:] = False

            if sl == 0 or align_mask_.sum() == 0 or res_batch_mask_.sum(
            ) == 0:
                continue

            prmsd_loss.append(
                bb_prmsd_l1(
                    out["dev_pred"],
                    flat_coords.detach(),
                    coords_label,
                    align_mask=align_mask_,
                    mask=res_batch_mask_,
                ))
        prmsd_loss = sum(prmsd_loss)

        losses = {
            "coord_loss": torch.mean(coords_loss),
            "bondlen_loss": torch.mean(bondlen_loss),
            "prmsd_loss": torch.mean(prmsd_loss)
        }
        for loss_name, loss in losses.items():
            if(torch.isnan(loss) or torch.isinf(loss)):
                logging.warning(f"{loss_name} loss is NaN. Skipping...")
                loss = loss.new_tensor(0., requires_grad=True)
            losses[loss_name] = loss
        return losses