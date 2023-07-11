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

import torch
from torch import nn

from solvent.data.protein_utils import _aatype_to_str_sequence
from solvent.utils.feats import atom14_to_atom37
from solvent.utils.rigid_utils import Rigid
from solvent.utils.tensor_utils import tensor_tree_map

from ..embedders import build_embedder
from ..folding import build_folding
from ..heads import build_head
from ..trunk import build_trunk
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class SSPF(nn.Module):
    """
    Meta architecture for Single Sequence based protein folding. 
    1. Get embedding via pre-trained protein language model
    2. embedding manipulation via trunk
    3. structure module for protein folding
    4. output head for predictions
    """
    
    def __init__(self, cfg):
        super().__init__()

        self.num_recycle = cfg.MODEL.NUM_RECYCLE
        self.seq_dim = cfg.MODEL.EMBEDDER.HIDDEN_DIM
        self.msa_dim = cfg.MODEL.TRUNK.MSA_DIM
        self.pair_dim = cfg.MODEL.TRUNK.PAIR_DIM
        self.offload_inference = cfg.MODEL.OFFLOAD_INFERENCE

        self.embedder = build_embedder(cfg)
        self.trunk = build_trunk(cfg)
        self.folding = build_folding(cfg)
        self.heads = build_head(cfg)

    @property
    def device(self):
        return next(self.embedder.parameters()).device
    
    def preprocess(self, batched_inputs):
        device = self.device
        batched_inputs = tensor_tree_map(
            lambda x: x.to(device),
            batched_inputs
        )

        aatypes = batched_inputs['aatype'][..., 0]
        seq_masks = batched_inputs['seq_mask'][..., 0]
        sequences = []
        for aa, mask in zip(aatypes, seq_masks):
            valid_aa = aa[mask==1]
            seq = _aatype_to_str_sequence(valid_aa)
            padding_num = (mask == 0).sum()
            seq += '<pad>' * padding_num
            sequences.append(seq)
        batched_inputs['seq'] = sequences

        return batched_inputs
    
    def iteration(self, feats, prevs, _recycle=True):
        # Primary output dictionary
        outputs = {}

        dtype = next(self.parameters()).dtype
        for k in feats:
            if(feats[k].dtype == torch.float32):
                feats[k] = feats[k].to(dtype=dtype)

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        outputs = self.trunk(
            outputs,
            feats, 
            prevs,
            inplace_safe=inplace_safe, 
            offload_inference=self.offload_inference
        )

        # Folding
        s = outputs["single"]
        m = outputs["msa"]

        outputs["sm"] = self.folding(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        # Save embeddings for use during the next recycling iteration

        # [*, N, C_m]
        m_1_prev = m[..., 0, :, :]

        # [*, N, N, C_z]
        z_prev = outputs["pair"]

        # [*, N, 3]
        x_prev = outputs["final_atom_positions"]
        
        return outputs, m_1_prev, z_prev, x_prev

    def forward(self, batched_inputs):
        batched_inputs = self.preprocess(batched_inputs)

        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev, x14_prev, r_prev = None, None, None, None, None
        prevs = [m_1_prev, z_prev, x_prev, x14_prev, r_prev]

        is_grad_enabled = torch.is_grad_enabled()
        num_iters = batched_inputs["aatype"].shape[-1]
        
        ## Get PLM embeddings and attentions
        batched_inputs = self.embedder(batched_inputs)
        del batched_inputs['seq']

        m = batched_inputs['representations']
        z = batched_inputs['attentions']
        del batched_inputs['representations']
        del batched_inputs['attentions']

        batched_inputs['msa'] = torch.cat(num_iters * [m.unsqueeze(-1)], dim=-1)
        batched_inputs['pair'] = torch.cat(num_iters * [z.unsqueeze(-1)], dim=-1)
        
        ## calculate folding from embeddings
        # Main recycling loop
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]
            feats = tensor_tree_map(fetch_cur_batch, batched_inputs)

            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.set_grad_enabled(is_grad_enabled and is_final_iter):
                if is_final_iter:
                    last_feats = feats
                    # Sidestep AMP bug (PyTorch issue #65766)
                    if torch.is_autocast_enabled():
                        torch.clear_autocast_cache()

                # Run the next iteration of the model
                inplace_safe = not (self.training or torch.is_grad_enabled())
                outputs, m_1_prev, z_prev, x_prev = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1),
                )

                if(not is_final_iter):
                    x14_prev = outputs['sm']['positions'][-1]
                    r_prev = Rigid.from_tensor_4x4(outputs['sm']['sidechain_frames'][-1])
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev, x14_prev, r_prev]
                    del m_1_prev, z_prev, x_prev, x14_prev, r_prev

        # Run auxiliary heads
        if self.training:
            losses = self.heads(outputs, last_feats)
            return losses
        else:
            results = self.heads(outputs, last_feats)
            return results