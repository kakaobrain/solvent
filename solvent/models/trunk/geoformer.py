# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from OmegaFold (https://github.com/HeliXonProtein/OmegaFold)
# Copyright 2022 HeliXon Limited

# modified from Modified from OpenFold (https://github.com/aqlaboratory/openfold)
# Copyright 2021 AlQuraishi Laboratory

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
import torch.nn as nn

from solvent.models.primitives import Linear
from solvent.models.trunk import Evoformer
from solvent.utils.feats import pseudo_beta_fn
from solvent.utils.rigid_utils import Rigid
from solvent.utils.tensor_utils import add
from ...common import residue_constants
from . import TRUNK, TRUNK_REGISTRY


class Val2ContBins(nn.Module):
    def __init__(self, min_bin, max_bin, no_bins):
        super(Val2ContBins, self).__init__()

        x_bin_size = (max_bin - min_bin) / (no_bins - 2)

        self.register_buffer(
            "x_offset", torch.linspace(
                min_bin - x_bin_size / 2,
                max_bin + x_bin_size / 2,
                no_bins
            ), persistent=False
        )
        self.coeff = -0.5 / ((x_bin_size * 0.2) ** 2)
        # `*0.5`: makes it not too blurred

    def forward(self, dist_x):  # (*)
        x_offset_shape = [1] * len(dist_x.size()) + [len(self.x_offset)]
        x = dist_x.unsqueeze(-1) - self.x_offset.view(*x_offset_shape)        
        x_norm = self.coeff * torch.pow(x, 2)
        x_norm = x_norm - x_norm.max(-1, keepdim=True)[0]
        logits = torch.softmax(x_norm, dim=-1)

        return logits

class StructEmbedder(nn.Module):
    """
    Encoder for pair wise atom distance without distance clamp
    but a sublinear-function with ord encoder.
    """

    def __init__(self, cfg):
        super(StructEmbedder, self).__init__()

        edge_dim = cfg.MODEL.TRUNK.PAIR_DIM
        rough_min_bin = cfg.MODEL.TRUNK.RECYCLE_EMB.MIN
        rough_max_bin = cfg.MODEL.TRUNK.RECYCLE_EMB.MAX
        rough_no_bins = cfg.MODEL.TRUNK.RECYCLE_EMB.NUM_BINS + 1
        dist_min_bin = cfg.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_MIN
        dist_max_bin = cfg.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_MAX
        dist_no_bins = cfg.MODEL.TRUNK.STRUCT_EMB.DIST_BINS_NUM
        pos_min_bin = cfg.MODEL.TRUNK.STRUCT_EMB.POS_BINS_MIN
        pos_max_bin = cfg.MODEL.TRUNK.STRUCT_EMB.POS_BINS_MAX
        pos_no_bins = cfg.MODEL.TRUNK.STRUCT_EMB.POS_BINS_NUM
        dim = cfg.MODEL.TRUNK.STRUCT_EMB.DIM

        self.rough_dist_bin = Val2ContBins(rough_min_bin, rough_max_bin, rough_no_bins)
        self.dist_bin = Val2ContBins(dist_min_bin, dist_max_bin, dist_no_bins)
        self.pos_bin = Val2ContBins(pos_min_bin, pos_max_bin, pos_no_bins)

        self.aa_embedding = nn.Embedding(21 * 21, embedding_dim=dim)

        frame_num = 1
        atom_num = 14

        self.dist_bin_embedding = Linear(dist_no_bins, dim)
        self.rough_dist_bin_embedding = Linear(rough_no_bins, dim)

        self.dist_bin_linear = Linear(atom_num * atom_num * dim, dim)
        self.rough_dist_bin_linear = Linear(atom_num * atom_num * dim, dim)

        self.pos_bin_embedding = Linear(pos_no_bins, dim)
        self.pos_linear = Linear(frame_num * atom_num * 3 * dim, dim)

        self.linear_z_weights = nn.Parameter(torch.zeros([dim, dim, edge_dim]))
        self.linear_z_bias = nn.Parameter(torch.zeros([edge_dim]))

    def forward(
            self,
            aatype1: torch.Tensor,
            aatype2: torch.Tensor,
            pos_a: torch.Tensor,
            mask_a: torch.Tensor,
            pos_b: torch.Tensor,
            mask_b: torch.Tensor,
            rigid: Rigid,
    ):
        pairwise_aatype = aatype1.unsqueeze(-1) * 21 + aatype1.unsqueeze(-2)
        d = torch.norm(
            pos_b[:, None, :, None] - pos_a[:, :, None, :, None],
            p=2, dim=-1, keepdim=False
        )
        d_mask = mask_b[:, None, :, None] * mask_a[:, :, None, :, None]
        d_mask = d_mask.unsqueeze(-1)
        local_mask = torch.mul(
            mask_b[:, None, :], mask_b[:, :, None]
        )
        local_mask = local_mask.unsqueeze(-1)

        local_vec = rigid[:, :, None, :, None].invert_apply(pos_b[:, None, :, None, :, :])
        local_vec = local_vec.mean(dim=-3)

        return self._sharded_compute(
            pairwise_aatype, d, local_vec, d_mask, local_mask
        )

    def _sharded_compute(
            self,
            pairwise_fasta: torch.Tensor,
            d: torch.Tensor,
            local_vec: torch.Tensor,
            d_mask: torch.Tensor,
            local_mask: torch.Tensor
    ) -> torch.Tensor:
        pairwise_fasta = self.aa_embedding(pairwise_fasta)
        d1 = self.rough_dist_bin(d)
        d2 = self.dist_bin(d)
        d3 = self.pos_bin(local_vec)

        d1 = self.rough_dist_bin_embedding(d1)
        d1 = d1 * d_mask
        d1 = self.rough_dist_bin_linear(d1.flatten(start_dim=-3))

        d2 = self.dist_bin_embedding(d2)
        d2 = d2 * d_mask
        d2 = self.dist_bin_linear(d2.flatten(start_dim=-3))

        d3 = self.pos_bin_embedding(d3)
        d3 = d3 * (local_mask.unsqueeze(-1))
        d3 = self.pos_linear(d3.flatten(start_dim=-3))

        final_d = d1 + d2 + d3  # + d4
        O = torch.einsum('...sdi,...sdj->...sdij', pairwise_fasta, final_d)
        Z = torch.einsum(
            '...sdij,ijh->...sdh', O, self.linear_z_weights
        ) + self.linear_z_bias
        return Z

class PairStructEmbedder(StructEmbedder):
    def forward(
            self,
            aatype: torch.Tensor,
            pos: torch.Tensor,
            pos_mask: torch.Tensor,
            rigid: Rigid,
    ):
        return super(PairStructEmbedder, self).forward(
            aatype1=aatype, aatype2=aatype,
            pos_a=pos, pos_b=pos,
            mask_a=pos_mask, mask_b=pos_mask,
            rigid=rigid
        )


@TRUNK_REGISTRY.register()
class GeoformerLite(Evoformer):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.embed_struct = cfg.MODEL.TRUNK.STRUCT_EMB.ENABLED
        if self.embed_struct:
            self.struct_embedder = PairStructEmbedder(cfg)

    def forward(self, outputs, feats, prevs, inplace_safe=False, offload_inference=False):
        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n = feats["target_feat"].shape[-2]
        n_seq = feats["msa_feat"].shape[-3]
        
        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]
        
        ## Initialize the MSA and pair representations
        m = feats['msa']
        z = feats['pair']

        # Unpack the recycling embeddings. Removing them from the list allows 
        # them to be freed further down in this function, saving memory
        m_1_prev, z_prev, x_prev, x14_prev, r_prev = reversed([prevs.pop() for _ in range(5)])

        # Initialize the recycling embeddings, if needs be 
        if None in [m_1_prev, z_prev, x_prev]:
            # [*, N, C_m]
            m_1_prev = m.new_zeros(
                (*batch_dims, n, self.msa_dim),
                requires_grad=False,
            )

            # [*, N, N, C_z]
            z_prev = z.new_zeros(
                (*batch_dims, n, n, self.pair_dim),
                requires_grad=False,
            )

            # [*, N, 3]
            x_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.atom_type_num, 3),
                requires_grad=False,
            )

            # [*, N, 3]
            x14_prev = z.new_zeros(
                (*batch_dims, n, residue_constants.restype_atom14_to_rigid_group.shape[1], 3),
                requires_grad=False,
            )
            
            b, seq_len = m.squeeze(1).shape[:-1]
            r_prev = Rigid.identity(
                # m.squeeze(1).shape[:-1], 
                (b, seq_len, 8),
                m.dtype, 
                m.device, 
                self.training,
                fmt="quat",
            )
        
        # The recycling embedder is memory-intensive, so we offload first
        if(offload_inference and inplace_safe):
            m = m.cpu()
            z = z.cpu()

        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        # m_1_prev_emb: [*, N, C_m]
        # z_prev_emb: [*, N, N, C_z]
        m_1_prev_emb, z_prev_emb = self.recycling_embedder(
            m_1_prev,
            z_prev,
            x_prev,
            inplace_safe=inplace_safe,
        )

        if(offload_inference and inplace_safe):
            m = m.to(m_1_prev_emb.device)
            z = z.to(z_prev.device)

        # [*, S_c, N, C_m]
        m[..., 0, :, :] += m_1_prev_emb

        # [*, N, N, C_z]
        z = add(z, z_prev_emb, inplace=inplace_safe)
        
        # Deletions like these become significant for inference with large N,
        # where they free unused tensors and remove references to others such
        # that they can be offloaded later
        del m_1_prev, z_prev, x_prev, m_1_prev_emb, z_prev_emb
        
        # struct embedder
        if self.embed_struct:
            if not self.training:
                # force cpu computation when inference
                # because it consume large GPU memory
                orig_device = m.device
                # m = m.cpu()
                z = z.cpu()

                r_prev = Rigid.from_tensor_7(r_prev.to_tensor_7().cpu())
                struct_emb = self.struct_embedder.to('cpu')(
                    feats["aatype"].cpu(),
                    x14_prev.cpu(),
                    feats["atom14_atom_exists"].cpu(),
                    r_prev,
                )
                z = add(z, struct_emb, inplace=inplace_safe)
                
                # m = m.to(orig_device)
                z = z.to(orig_device)
                self.struct_embedder.to(orig_device)

            else:
                struct_emb = self.struct_embedder(
                    feats["aatype"],
                    x14_prev,
                    feats["atom14_atom_exists"],
                    r_prev,
                )
                z = add(z, struct_emb, inplace=inplace_safe)            

            del x14_prev, r_prev, struct_emb

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if(offload_inference):
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.chunk_size,
                use_lma=self.use_lma,
                _mask_trans=self._mask_trans,
            )
    
            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.chunk_size,
                use_lma=self.use_lma,
                use_flash=self.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        return outputs