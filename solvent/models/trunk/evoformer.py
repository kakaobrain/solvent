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

import sys
from functools import partial
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from detectron2.config import configurable

from solvent.layers import (
    DropoutColumnwise, DropoutRowwise,
    MSAColumnAttention, MSAColumnGlobalAttention,
    MSARowAttentionWithPairBias, OuterProductMean,
    PairTransition, TriangleAttention,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing
)
from solvent.models.primitives import LayerNorm, Linear
from solvent.utils.checkpointing import checkpoint_blocks, get_checkpoint_fn
from solvent.utils.chunk_utils import ChunkSizeTuner, chunk_layer
from solvent.utils.feats import pseudo_beta_fn
from solvent.utils.tensor_utils import add
from ...common import residue_constants
from . import TRUNK, TRUNK_REGISTRY


class RecyclingEmbedder(nn.Module):
    """
    Embeds the output of an iteration of the model for recycling.

    Implements Algorithm 32.
    """
    @configurable
    def __init__(
        self,
        c_m: int,
        c_z: int,
        min_bin: float,
        max_bin: float,
        no_bins: int,
        inf: float = 1e8,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair embedding channel dimension
            min_bin:
                Smallest distogram bin (Angstroms)
            max_bin:
                Largest distogram bin (Angstroms)
            no_bins:
                Number of distogram bins
        """
        super(RecyclingEmbedder, self).__init__()

        self.c_m = c_m
        self.c_z = c_z
        self.min_bin = min_bin
        self.max_bin = max_bin
        self.no_bins = no_bins
        self.inf = inf

        self.linear = Linear(self.no_bins, self.c_z)
        self.layer_norm_m = LayerNorm(self.c_m)
        self.layer_norm_z = LayerNorm(self.c_z)
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "c_m": cfg.MODEL.TRUNK.MSA_DIM,
            "c_z": cfg.MODEL.TRUNK.PAIR_DIM,
            "min_bin": cfg.MODEL.TRUNK.RECYCLE_EMB.MIN,
            "max_bin": cfg.MODEL.TRUNK.RECYCLE_EMB.MAX,
            "no_bins": cfg.MODEL.TRUNK.RECYCLE_EMB.NUM_BINS,
        }

    def forward(
        self,
        m: torch.Tensor,
        z: torch.Tensor,
        x: torch.Tensor,
        inplace_safe: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                First row of the MSA embedding. [*, N_res, C_m]
            z:
                [*, N_res, N_res, C_z] pair embedding
            x:
                [*, N_res, 3] predicted C_beta coordinates
        Returns:
            m:
                [*, N_res, C_m] MSA embedding update
            z:
                [*, N_res, N_res, C_z] pair embedding update
        """
        # [*, N, C_m]
        m_update = self.layer_norm_m(m)
        if(inplace_safe):
            m.copy_(m_update)
            m_update = m

        # [*, N, N, C_z]
        z_update = self.layer_norm_z(z)
        if(inplace_safe):
            z.copy_(z_update)
            z_update = z

        # This squared method might become problematic in FP16 mode.
        bins = torch.linspace(
            self.min_bin,
            self.max_bin,
            self.no_bins,
            dtype=x.dtype,
            device=x.device,
            requires_grad=False,
        )
        squared_bins = bins ** 2
        upper = torch.cat(
            [squared_bins[1:], squared_bins.new_tensor([self.inf])], dim=-1
        )
        d = torch.sum(
            (x[..., None, :] - x[..., None, :, :]) ** 2, dim=-1, keepdims=True
        )

        # [*, N, N, no_bins]
        d = ((d > squared_bins) * (d < upper)).type(x.dtype)

        # [*, N, N, C_z]
        d = self.linear(d)
        z_update = add(z_update, d, inplace_safe)

        return m_update, z_update


class MSATransition(nn.Module):
    """
    Feed-forward network applied to MSA activations after attention.

    Implements Algorithm 9
    """
    def __init__(self, c_m, n):
        """
        Args:
            c_m:
                MSA channel dimension
            n:
                Factor multiplied to c_m to obtain the hidden channel
                dimension
        """
        super(MSATransition, self).__init__()

        self.c_m = c_m
        self.n = n

        self.layer_norm = LayerNorm(self.c_m)
        self.linear_1 = Linear(self.c_m, self.n * self.c_m, init="relu")
        self.relu = nn.ReLU()
        self.linear_2 = Linear(self.n * self.c_m, self.c_m, init="final")

    def _transition(self, m, mask):
        m = self.layer_norm(m)
        m = self.linear_1(m)
        m = self.relu(m)
        m = self.linear_2(m) * mask
        return m

    @torch.jit.ignore
    def _chunk(self,
        m: torch.Tensor,
        mask: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
         return chunk_layer(
             self._transition,
             {"m": m, "mask": mask},
             chunk_size=chunk_size,
             no_batch_dims=len(m.shape[:-2]),
         )


    def forward(
        self,
        m: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA activation
            mask:
                [*, N_seq, N_res, C_m] MSA mask
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA activation update
        """
        # DISCREPANCY: DeepMind forgets to apply the MSA mask here.
        if mask is None:
            mask = m.new_ones(m.shape[:-1])

        mask = mask.unsqueeze(-1)

        if chunk_size is not None:
            m = self._chunk(m, mask, chunk_size)
        else:
            m = self._transition(m, mask)

        return m


class EvoformerBlockCore(nn.Module):
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        pair_dropout: float,
        inf: float,
        eps: float,
        _is_extra_msa_stack: bool = False,
    ):
        super(EvoformerBlockCore, self).__init__()

        self.msa_transition = MSATransition(
            c_m=c_m,
            n=transition_n,
        )

        self.outer_product_mean = OuterProductMean(
            c_m,
            c_z,
            c_hidden_opm,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(
            c_z,
            c_hidden_mul,
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            c_z,
            c_hidden_mul,
        )

        self.tri_att_start = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )
        self.tri_att_end = TriangleAttention(
            c_z,
            c_hidden_pair_att,
            no_heads_pair,
            inf=inf,
        )

        self.pair_transition = PairTransition(
            c_z,
            transition_n,
        )

        self.ps_dropout_row_layer = DropoutRowwise(pair_dropout)

    def forward(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]: 
        # DeepMind doesn't mask these transitions in the source, so _mask_trans
        # should be disabled to better approximate the exact activations of
        # the original.
        msa_trans_mask = msa_mask if _mask_trans else None
        pair_trans_mask = pair_mask if _mask_trans else None
      
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        m, z = input_tensors
        
        m = add(
            m,
            self.msa_transition(
                m, mask=msa_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        ) 

        if(_offload_inference and inplace_safe):
            del m, z
            assert(sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[1] = input_tensors[1].cpu()
            torch.cuda.empty_cache()
            m, z = input_tensors 

        opm = self.outer_product_mean(
            m, mask=msa_mask, chunk_size=chunk_size, inplace_safe=inplace_safe
        )

        if(_offload_inference and inplace_safe):
            del m, z
            assert(sys.getrefcount(input_tensors[0]) == 2)
            input_tensors[0] = input_tensors[0].cpu()
            input_tensors[1] = input_tensors[1].to(opm.device)
            m, z = input_tensors

        z = add(z, opm, inplace=inplace_safe)
        del opm

        tmu_update = self.tri_mul_out(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if(not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
        
        del tmu_update

        tmu_update = self.tri_mul_in(
            z,
            mask=pair_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=True,
        )
        if(not inplace_safe):
            z = z + self.ps_dropout_row_layer(tmu_update)
        else:
            z = tmu_update
       
        del tmu_update

        z = add(z, 
            self.ps_dropout_row_layer(
                self.tri_att_start(
                    z, 
                    mask=pair_mask, 
                    chunk_size=_attn_chunk_size, 
                    use_memory_efficient_kernel=False,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = add(z,
            self.ps_dropout_row_layer(
                self.tri_att_end(
                    z,
                    mask=pair_mask.transpose(-1, -2),
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_lma=use_lma,
                    inplace_safe=inplace_safe,
                )
            ),
            inplace=inplace_safe,
        )

        z = z.transpose(-2, -3)
        
        if(inplace_safe):
            input_tensors[1] = z.contiguous()
            z = input_tensors[1]

        z = add(z,
            self.pair_transition(
                z, mask=pair_trans_mask, chunk_size=chunk_size,
            ),
            inplace=inplace_safe,
        )

        if(_offload_inference and inplace_safe):
            device = z.device
            del m, z
            assert(sys.getrefcount(input_tensors[0]) == 2)
            assert(sys.getrefcount(input_tensors[1]) == 2)
            input_tensors[0] = input_tensors[0].to(device)
            input_tensors[1] = input_tensors[1].to(device)
            m, z = input_tensors

        return m, z


class EvoformerBlock(nn.Module):
    def __init__(self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        no_heads_msa: int,
        no_heads_pair: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        inf: float,
        eps: float,
        use_msa: bool,
    ):
        super(EvoformerBlock, self).__init__()

        self.msa_att_row = MSARowAttentionWithPairBias(
            c_m=c_m,
            c_z=c_z,
            c_hidden=c_hidden_msa_att,
            no_heads=no_heads_msa,
            inf=inf,
        )
        if use_msa:
            self.msa_att_col = MSAColumnAttention(
                c_m,
                c_hidden_msa_att,
                no_heads_msa,
                inf=inf,
            )

        self.msa_dropout_layer = DropoutRowwise(msa_dropout)

        self.core = EvoformerBlockCore(
            c_m=c_m,
            c_z=c_z,
            c_hidden_opm=c_hidden_opm,
            c_hidden_mul=c_hidden_mul,
            c_hidden_pair_att=c_hidden_pair_att,
            no_heads_msa=no_heads_msa,
            no_heads_pair=no_heads_pair,
            transition_n=transition_n,
            pair_dropout=pair_dropout,
            inf=inf,
            eps=eps,
        )

        self.use_msa = use_msa

    def forward(self,
        m: Optional[torch.Tensor],
        z: Optional[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: Optional[int] = None,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
        _attn_chunk_size: Optional[int] = None,
        _offload_inference: bool = False,
        _offloadable_inputs: Optional[Sequence[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if(_attn_chunk_size is None):
            _attn_chunk_size = chunk_size

        if(_offload_inference and inplace_safe):
            input_tensors = _offloadable_inputs
            del _offloadable_inputs
        else:
            input_tensors = [m, z]

        m, z = input_tensors

        m = add(m, 
            self.msa_dropout_layer(
                self.msa_att_row(
                    m, 
                    z=z, 
                    mask=msa_mask, 
                    chunk_size=_attn_chunk_size,
                    use_memory_efficient_kernel=False,
                    use_lma=use_lma,
                )
            ),
            inplace=inplace_safe,
        )
        
        if self.use_msa:
            m = add(m, 
                self.msa_att_col(
                    m, 
                    mask=msa_mask, 
                    chunk_size=chunk_size,
                    use_lma=use_lma,
                    use_flash=use_flash,
                ),
                inplace=inplace_safe,
            )

        if(not inplace_safe):
            input_tensors = [m, input_tensors[1]]
        
        del m, z

        m, z = self.core(
            input_tensors, 
            msa_mask=msa_mask, 
            pair_mask=pair_mask, 
            chunk_size=chunk_size, 
            use_lma=use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=_mask_trans,
            _attn_chunk_size=_attn_chunk_size,
            _offload_inference=_offload_inference,
        )

        return m, z

class EvoformerStack(nn.Module):
    """
    Main Evoformer trunk.

    Implements Algorithm 6.
    """
    @configurable
    def __init__(
        self,
        c_m: int,
        c_z: int,
        c_hidden_msa_att: int,
        c_hidden_opm: int,
        c_hidden_mul: int,
        c_hidden_pair_att: int,
        c_s: int,
        no_heads_msa: int,
        no_heads_pair: int,
        no_blocks: int,
        transition_n: int,
        msa_dropout: float,
        pair_dropout: float,
        blocks_per_ckpt: int,
        inf: float = 5e4,
        eps: float = 1e-8,
        clear_cache_between_blocks: bool = False, 
        tune_chunk_size: bool = False,
        use_msa: bool = False,
        **kwargs,
    ):
        """
        Args:
            c_m:
                MSA channel dimension
            c_z:
                Pair channel dimension
            c_hidden_msa_att:
                Hidden dimension in MSA attention
            c_hidden_opm:
                Hidden dimension in outer product mean module
            c_hidden_mul:
                Hidden dimension in multiplicative updates
            c_hidden_pair_att:
                Hidden dimension in triangular attention
            c_s:
                Channel dimension of the output "single" embedding
            no_heads_msa:
                Number of heads used for MSA attention
            no_heads_pair:
                Number of heads used for pair attention
            no_blocks:
                Number of Evoformer blocks in the stack
            transition_n:
                Factor by which to multiply c_m to obtain the MSATransition
                hidden dimension
            msa_dropout:
                Dropout rate for MSA activations
            pair_dropout:
                Dropout used for pair activations
            blocks_per_ckpt:
                Number of Evoformer blocks in each activation checkpoint
            clear_cache_between_blocks:
                Whether to clear CUDA's GPU memory cache between blocks of the
                stack. Slows down each block but can reduce fragmentation
            tune_chunk_size:
                Whether to dynamically tune the module's chunk size
        """
        super(EvoformerStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt
        self.clear_cache_between_blocks = clear_cache_between_blocks

        if no_blocks > 0:
            self.blocks = nn.ModuleList()
            for _ in range(no_blocks):
                block = EvoformerBlock(
                    c_m=c_m,
                    c_z=c_z,
                    c_hidden_msa_att=c_hidden_msa_att,
                    c_hidden_opm=c_hidden_opm,
                    c_hidden_mul=c_hidden_mul,
                    c_hidden_pair_att=c_hidden_pair_att,
                    no_heads_msa=no_heads_msa,
                    no_heads_pair=no_heads_pair,
                    transition_n=transition_n,
                    msa_dropout=msa_dropout,
                    pair_dropout=pair_dropout,
                    inf=inf,
                    eps=eps,
                    use_msa=use_msa,
                )
                self.blocks.append(block)
        else:
            self.linear_m = Linear(c_m, c_m)
            self.linear_z = Linear(c_z, c_z)

        self.linear = Linear(c_m, c_s)

        self.no_blocks = no_blocks
        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if(tune_chunk_size):
            self.chunk_size_tuner = ChunkSizeTuner()

    @classmethod
    def from_config(cls, cfg):
        return {
            "c_m": cfg.MODEL.TRUNK.MSA_DIM,
            "c_z": cfg.MODEL.TRUNK.PAIR_DIM,
            "c_hidden_msa_att": cfg.MODEL.TRUNK.ATTENTION.DIM,
            "c_hidden_opm": cfg.MODEL.TRUNK.OUTERPRODUCT.DIM,
            "c_hidden_mul": cfg.MODEL.TRUNK.TRIANGLE.DIM,
            "c_hidden_pair_att": cfg.MODEL.TRUNK.TRIANGLE.ATTN_DIM,
            "c_s": cfg.MODEL.TRUNK.SINGLE_DIM,
            "no_heads_msa": cfg.MODEL.TRUNK.ATTENTION.DIM,
            "no_heads_pair": cfg.MODEL.TRUNK.TRIANGLE.ATTN_NUM_HEADS,
            "no_blocks": cfg.MODEL.TRUNK.NUM_BLOCKS,
            "transition_n": cfg.MODEL.TRUNK.TRANSITION.INTERMEDIATE_FACTOR,
            "msa_dropout": cfg.MODEL.TRUNK.DROPOUT_RATE,
            "pair_dropout": cfg.MODEL.TRUNK.TRIANGLE.DROPOUT_RATE,
            "blocks_per_ckpt": cfg.MODEL.BLOCKS_PER_CKPT,
            "clear_cache_between_blocks": cfg.MODEL.TRUNK.CLEAR_CACHE_BETWEEN_BLOCKS,
            "tune_chunk_size": cfg.MODEL.TUNE_CHUNK_SIZE,
            "use_msa": cfg.MODEL.TRUNK.USE_MSA,
        }

    def _prep_blocks(self, 
        m: torch.Tensor, 
        z: torch.Tensor, 
        chunk_size: int,
        use_lma: bool,
        use_flash: bool,
        msa_mask: Optional[torch.Tensor],
        pair_mask: Optional[torch.Tensor],
        inplace_safe: bool,
        _mask_trans: bool,
    ):
        blocks = [
            partial(
                b,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                chunk_size=chunk_size,
                use_lma=use_lma,
                use_flash=use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            for b in self.blocks
        ]

        if(self.clear_cache_between_blocks):
            def block_with_cache_clear(block, *args, **kwargs):
                torch.cuda.empty_cache()
                return block(*args, **kwargs)

            blocks = [partial(block_with_cache_clear, b) for b in blocks]

        if(chunk_size is not None and self.chunk_size_tuner is not None):
            assert(not self.training)
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                # We don't want to write in-place during chunk tuning runs
                args=(m.clone(), z.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b, 
                    chunk_size=tuned_chunk_size,
                    # A temporary measure to address torch's occasional
                    # inability to allocate large tensors
                    _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                ) for b in blocks
            ]

        return blocks

    def _forward_offload(self,
        input_tensors: Sequence[torch.Tensor],
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
        use_flash: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert(not (self.training or torch.is_grad_enabled()))
        
        if self.no_blocks > 0:
            blocks = self._prep_blocks(
                # We are very careful not to create references to these tensors in
                # this function
                m=input_tensors[0],
                z=input_tensors[1],
                chunk_size=chunk_size,
                use_lma=use_lma,
                use_flash=use_flash,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                inplace_safe=True,
                _mask_trans=_mask_trans,
            )

            for b in blocks:
                m, z = b(
                    None, 
                    None, 
                    _offload_inference=True,
                    _offloadable_inputs=input_tensors,
                )
                input_tensors[0] = m
                input_tensors[1] = z
                del m, z
            
            m, z = input_tensors
        else:
            m = self.linear_m(input_tensors[0])
            z = self.linear_z(input_tensors[1])
        
        s = self.linear(m[..., 0, :, :])
        
        return m, z, s

    def forward(self,
        m: torch.Tensor,
        z: torch.Tensor,
        msa_mask: torch.Tensor,
        pair_mask: torch.Tensor,
        chunk_size: int,
        use_lma: bool = False,
        use_flash: bool = False,
        inplace_safe: bool = False,
        _mask_trans: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            msa_mask:
                [*, N_seq, N_res] MSA mask
            pair_mask:
                [*, N_res, N_res] pair mask
            chunk_size: 
                Inference-time subbatch size. Acts as a minimum if 
                self.tune_chunk_size is True
            use_lma: Whether to use low-memory attention during inference
            use_flash: 
                Whether to use FlashAttention where possible. Mutually 
                exclusive with use_lma.
        Returns:
            m:
                [*, N_seq, N_res, C_m] MSA embedding
            z:
                [*, N_res, N_res, C_z] pair embedding
            s:
                [*, N_res, C_s] single embedding (or None if extra MSA stack)
        """ 
        if self.no_blocks > 0:
            blocks = self._prep_blocks(
                m=m,
                z=z,
                chunk_size=chunk_size,
                use_lma=use_lma,
                use_flash=use_flash,
                msa_mask=msa_mask,
                pair_mask=pair_mask,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )

            blocks_per_ckpt = self.blocks_per_ckpt
            if(not torch.is_grad_enabled()):
                blocks_per_ckpt = None
            
            m, z = checkpoint_blocks(
                blocks,
                args=(m, z),
                blocks_per_ckpt=blocks_per_ckpt,
            )
        else:
            m = self.linear_m(m)
            z = self.linear_z(z)

        s = self.linear(m[..., 0, :, :])

        return m, z, s

@TRUNK_REGISTRY.register()
class Evoformer(TRUNK):
    def __init__(self, cfg):
        super().__init__()

        self.msa_dim = cfg.MODEL.TRUNK.MSA_DIM
        self.pair_dim = cfg.MODEL.TRUNK.PAIR_DIM
        self.chunk_size = cfg.MODEL.CHUNK_SIZE
        self.use_lma = cfg.MODEL.USE_LMA
        self.use_flash = cfg.MODEL.USE_FLASH
        self._mask_trans = cfg.MODEL._MASK_TRANS

        self.recycling_embedder = RecyclingEmbedder(cfg)
        self.evoformer = EvoformerStack(cfg)

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
        m_1_prev, z_prev, x_prev, _, _ = reversed([prevs.pop() for _ in range(5)])

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

        x_prev = pseudo_beta_fn(
            feats["aatype"], x_prev, None
        ).to(dtype=z.dtype)

        # The recycling embedder is memory-intensive, so we offload first
        if(offload_inference and inplace_safe):
            m = m.cpu()
            z = z.cpu()

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