# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from IgFold (https://github.com/Graylab/IgFold)
# Copyright 2022 The Johns Hopkins University
# https://github.com/Graylab/IgFold/blob/main/LICENSE.md

# Modified from OpenFold (https://github.com/aqlaboratory/openfold)
# Copyright 2021 AlQuraishi Laboratory

import torch
from torch import nn

from solvent.common import residue_constants
from solvent.data.protein_utils import _aatype_to_str_sequence
from solvent.utils.feats import atom14_to_atom37
from solvent.utils.tensor_utils import (
    batched_gather, 
    dict_multimap,
    tensor_tree_map
)
from ..embedders import build_embedder
from ..folding import build_folding
from ..heads import build_head
from ..trunk import build_trunk
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class IGFold(nn.Module):
    """
    Meta architecture for Single Sequence based protein folding. 
    1. Get embeeding via pre-trained protein language model
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
        # padding
        aatypes = batched_inputs['aatype'][..., -1:]
        seq_mask = batched_inputs['seq_mask'].bool()
        sequence = _aatype_to_str_sequence(aatypes.squeeze(0))
        sequence = " ".join(list(sequence))

        batched_inputs['aatype'] = aatypes
        batched_inputs['seq'] = [sequence]

        return batched_inputs

    def template(self, batched_inputs, use_template=False):
        aatypes = batched_inputs['aatype'][..., -1:]
        # templates
        if self.training:
            template_coords = batched_inputs['all_atom_positions'].clone()
            template_mask = batched_inputs['all_atom_mask'].clone()
            template_mask[template_coords.isnan().any(-2)] = False
            template_mask[template_coords.sum(-2) == 0] = False
            
            b, n, dim, _, num_recycle = template_coords.shape
            if use_template:
                stride = 20
                for i in range(0, template_coords.shape[1], stride):
                    start_p = torch.randint(0, stride, (1,))
                    length = torch.randint(1, 7, (1,))
                    temp_coords = template_coords[:, i:i+stride]
                    temp_coords[:, start_p:start_p+length] = 0
                    template_coords[:, i:i+stride] = temp_coords

                    temp_mask = template_mask[:, i:i+stride]
                    temp_mask[:, start_p:start_p+length] = 0
                    template_mask[:, i:i+stride] = temp_mask

                cdr_index = batched_inputs['cdr_index'][0, :, 0]
                template_coords[:, cdr_index] = 0
                template_mask[:, cdr_index] = 0

                batched_inputs["template_all_atom_positions"] = template_coords
                batched_inputs["template_all_atom_mask"] = template_mask
            else:
                batched_inputs["template_all_atom_positions"] = torch.zeros_like(template_coords)
                batched_inputs["template_all_atom_mask"] = torch.zeros_like(template_mask)
        else:
            b, seq_len, num_recycle = batched_inputs['aatype'].shape
            batched_inputs["template_all_atom_positions"] = torch.zeros((b, seq_len, 37, 3, num_recycle), device=aatypes.device)
            batched_inputs["template_all_atom_mask"] = torch.zeros((b, seq_len, 37, num_recycle), device=aatypes.device)

        temp_coords = batched_inputs["template_all_atom_positions"]
        temp_mask = batched_inputs["template_all_atom_mask"]
        for i, (tc, m) in enumerate(zip(temp_coords, temp_mask)):
            m = m.unsqueeze(-2).repeat(1, 1, 3, 1)
            mean = (tc * m).mean(-2).unsqueeze(-2).repeat(1, 1, num_recycle, 1)
            temp_coords[i] -= mean
        batched_inputs['template_all_atom_positions'] = temp_coords

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

        outputs = self.folding(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.offload_inference,
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"][..., :5]
        
        return outputs

    def forward(self, batched_inputs):
        # only support batch size = 1 for igfold
        assert batched_inputs["aatype"].shape[0] == 1
        batch_size = 1

        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev, x14_prev, r_prev = None, None, None, None, None
        prevs = [m_1_prev, z_prev, x_prev, x14_prev, r_prev]

        is_grad_enabled = torch.is_grad_enabled()
        num_iters = batched_inputs["aatype"].shape[-1]
        
        ## Get PLM embeddings and attentions
        device = self.device
        batched_inputs = tensor_tree_map(
            lambda x: x.to(device),
            batched_inputs
        )
        
        orig_aatype = batched_inputs['aatype'].clone()
        orig_seq_mask = batched_inputs['seq_mask'].clone()
        
        chain_index = batched_inputs['chain_index'][..., -1:]
        valid_aa = batched_inputs['seq_mask'][..., -1:]
        valid_chain_idx = torch.logical_and(chain_index, valid_aa)
        chain_ids = chain_index[valid_chain_idx].unique()   # 7, 11
        
        use_template = torch.randint(0, 2, (1,))
        batched_inputs = self.template(batched_inputs, use_template)

        ms = []
        zs = []
        seq_lens = []
        for chain_id in chain_ids:
            masking = (chain_index == chain_id).int()
            seq_lens.append(masking.sum())
            batched_inputs['seq_mask'] = masking
            batched_inputs['aatype'] = orig_aatype[:, masking[0, :, 0].bool(), :]
            batched_inputs = self.preprocess(batched_inputs)

            ## Get PLM embeddings and attentions
            batched_inputs = self.embedder(batched_inputs)
            del batched_inputs['seq']

            m = batched_inputs['representations']
            z = batched_inputs['attentions']
            del batched_inputs['representations']
            del batched_inputs['attentions']

            ms.append(m)
            zs.append(z)
        batched_inputs['aatype'] = orig_aatype
        batched_inputs['seq_mask'] = orig_seq_mask
        batched_inputs['masking_index'] = orig_seq_mask.unsqueeze(1)
        
        # merge heavy-light
        m = torch.cat(ms, dim=2)

        seq_len = sum(seq_lens)
        z = torch.zeros(
            (batch_size, seq_len, seq_len, zs[0].shape[-1]),
            device=self.device,
        )
        for i, (a, l) in enumerate(zip(zs, seq_lens)):
            cum_l = sum(seq_lens[:i])
            z[:, cum_l:cum_l + l, cum_l:cum_l + l, :] = a


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
                outputs  = self.iteration(
                    feats,
                    prevs,
                    _recycle=(num_iters > 1),
                )

                if(not is_final_iter):
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        # Run auxiliary heads
        if self.training:
            losses = self.heads(outputs, last_feats)
            return losses
        else:
            results = self.heads(outputs, last_feats)
            return results