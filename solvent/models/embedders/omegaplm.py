# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from OmegaFold (https://github.com/HeliXonProtein/OmegaFold)
# Copyright 2022 HeliXon Limited
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

import math
import numbers
import typing

import torch
from torch import nn

from ..primitives import LayerNorm, Linear, _naive_layernorm, _softmax
from .build import EMBEDDER_REGISTRY
from .embedder import EMBEDDER


def _attention(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: torch.Tensor,
        value: torch.Tensor,
        bias: torch.Tensor,
        return_edge: bool,
        edge_reduction: str,
        edge_reduction_dim: int
) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
    """Normal attention

    Args:
        query: positive tensor of shape (*_q, dim_qk)
        key: positive tensor of shape (*_k, dim_qk)
        scale: the scaling of logits
        value: tensor of shape (*_k, dim_v)
        bias: the bias acting as either mask or relative positional encoding
        return_edge: if to return the logits of attention

    Returns:
        The aggregated tensor of shape (*_q, dim_v)

    """
    logits = torch.einsum("...id, ...jd -> ...ij", query * scale, key)
    logits.add_(bias)
    attn = _softmax(logits, dim=-1, in_place=False)
    out = torch.einsum("...ij, ...jd -> ...id", attn, value)
    if return_edge:
        attn = getattr(attn, edge_reduction)(dim=edge_reduction_dim)
        return out, attn
    else:
        return out, None


def attention(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: typing.Union[torch.Tensor, float],
        value: torch.Tensor,
        bias: torch.Tensor,
        subbatch_size: typing.Optional[int] = None,
        *,
        return_edge: bool = False,
        edge_reduction: str = 'sum',
        edge_reduction_dim: int = 0,
) -> typing.Tuple[torch.Tensor, typing.Tuple[torch.Tensor]]:
    """Computes attention with q, k , v

    Args:
        query: positive tensor of shape (*_q, dim_qk)
        key: positive tensor of shape (*_k, dim_qk)
        scale: the scaling of logits
        value: tensor of shape (*_k, dim_v)
        bias: the bias acting as either mask or relative positional encoding
        subbatch_size: the subbatch size to split the computation into
        return_edge: if to return the logits
        edge_reduction:
        edge_reduction_dim:

    Returns:
        The aggregated tensor of shape (*_q, dim_v)

    """
    q_length, k_length, v_dim = query.shape[-2], key.shape[-2], value.shape[-1]
    subbatch_size = subbatch_size or q_length

    batch_shape = list(query.shape[:-2])
    factory_kwargs = nn.factory_kwargs(
        {"device": query.device, "dtype": query.dtype}
    )
    output = torch.empty(*batch_shape, q_length, v_dim, **factory_kwargs)
    if return_edge:
        batch_shape.pop(edge_reduction_dim + 2)
        attns = torch.empty(
            *batch_shape, q_length, k_length, **factory_kwargs
        )
    else:
        attns = None

    for i, q_i in enumerate(query.split(subbatch_size, dim=-2)):
        start, end = i * subbatch_size, (i + 1) * subbatch_size,
        if bias.shape[-2] != q_length:
            b_i = bias
        else:
            b_i = bias[..., start:end, :]

        res, attn = _attention(
            q_i, key, scale, value, b_i, return_edge,
            edge_reduction, edge_reduction_dim
        )
        output[..., start:end, :] = res
        if return_edge:
            attns[..., start:end, :] = attn

    return output, attns

def mask2bias(mask: torch.Tensor, *, inf: float = 1e9) -> torch.Tensor:
    """Convert mask to attention bias

    Args:
        mask: the mask to convert to bias representation
        inf: the floating point number to represent infinity

    Returns:
        bias representation for masking in attention
        1e9 - 8103
    """
    return mask.float().sub(1).mul(inf)

def _get_qk_scaling(num_res, attn_dim):
    return num_res.clamp(min=4e-5).log() / (math.log(512) * attn_dim ** 0.5)

def _get_pos(shape,device,dtype,seq_dim):
    """Get the position of the tokens given

    Args:
        shape: the shape of the tensor to be applied with RoPE
        device: the device on which the tensor reside
        dtype: the datatype of the tensor
        seq_dim: dimensions of the tensor that reference the sequence length

    Returns:
        The position tensor of the shape from ~shape indexed by seq_dim

    """
    spatial_shape = [shape[i] for i in seq_dim]
    total_len = 1
    for i in spatial_shape:
        total_len *= i
    position = torch.arange(total_len, dtype=dtype, device=device)
    position = position.reshape(*spatial_shape).contiguous()

    return position # position info. vector

def _apply_embed(inputs,sin,cos,seq_dim):
    """Applies RoPE to ~inputs

    Args:
        inputs: the tensor to which RoPE is applied, the dimensions indexed by
            ~seq_dim indicates the spatial dimensions
        sin: the sine tensor that constitutes parts of the RoPE,
            of spatial shape + vector dimension
        cos: the cosine tensor that constitutes parts of the RoPE,
            of spatial shape + vector dimension
        seq_dim: the dimensions indicating the spatial dimensions,
            must be consecutive

    Returns:
        tensor with RoPE applied.

    """
    gaps = [
        (seq_dim[i + 1] - seq_dim[i]) == 1 for i in range(len(seq_dim) - 1)
    ]
    if len(gaps) > 0:
        if not all(gaps):
            raise ValueError(f"seq_dim must be consecutive, but got {seq_dim}")

    # Align dimensions of sine and cosine
    seq_dim = sorted(seq_dim)
    end = seq_dim[-1]
    for _ in range(seq_dim[0]):
        sin = sin.unsqueeze(0)
        cos = cos.unsqueeze(0)
        end += 1

    for _ in range(end, inputs.ndim - 1):
        sin = sin.unsqueeze(_)
        cos = cos.unsqueeze(_)

    # Apply RoPE
    x1, x2 = torch.split(inputs, inputs.shape[-1] // 2, dim=-1)

    # dimension matching
    if x1.ndim == 5:
        cos = cos.unsqueeze(3)
        sin = sin.unsqueeze(3)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class RelPosEmbedder(nn.Embedding):
    """
        Compute the relative positional embedding, this is the same algorithm in
        Jumper et al. (2021) Suppl. Alg. 4 "relpos"
    """

    def forward(self, num_res: int) -> torch.Tensor:
        """

        Args:
            num_res: number of residues in input sequence.

        Returns:

        """
        idx = torch.arange(num_res, device=next(self.parameters()).device)
        one_side = self.num_embeddings // 2
        idx = (idx[None, :] - idx[:, None]).clamp(-one_side, one_side)
        idx = idx + one_side
        return super(RelPosEmbedder, self).forward(idx)


class RoPE(nn.Module):
    """The RoPE module
    Attributes:
        input_dim: the dimension of the input vectors.
    """

    def __init__(self, input_dim: int) -> None:
        super(RoPE, self).__init__()
        if input_dim % 2 != 0:
            raise ValueError(
                f"Input dimension for RoPE must be a multiple of 2,"
                f" but got {input_dim}"
            )
        self.input_dim = input_dim
        self.half_size = input_dim // 2
        '''
                                                                 [0,1,2,3,4,5,...,10]
            torch.arange(self.half_size, dtype=torch.float32) -> [index0, index1, ..., index_half]
            freq_seq = -freq_seq.div(float(self.half_size))   -> [0/11, 1/11,...,10/11] and inverse
            each element power to 10000
        '''
        freq_seq = torch.arange(self.half_size, dtype=torch.float32) # [0,1,2,3,4,5,...,10]
        freq_seq = -freq_seq.div(float(self.half_size)) # [0/11, 1/11,...,10/11] and inverse
        self.register_buffer(
            "inv_freq", torch.pow(10000., freq_seq), persistent=False
        )

    def forward(self, tensor, seq_dim) -> torch.Tensor:
        """
        Args:
            tensor: the tensor to apply rope onto
            seq_dim: the dimension that represents the sequence dimension
        """
        if isinstance(seq_dim, int):
            seq_dim = [seq_dim, ]
        sin, cos = self._compute_sin_cos(tensor, seq_dim)

        return _apply_embed(tensor, sin, cos, seq_dim)

    def _compute_sin_cos(self, tensor, seq_dim):
        """Compute sine and cosine tensors
        Args:
            tensor: the tensors to apply RoPE to
            seq_dim: the dimension indices of the spatial dimensions
        Returns:
            A tuple of tensors where the first one is the sine tensor
                and the second one is the cosine tensor
        """
        position = _get_pos(tensor.shape, tensor.device, tensor.dtype, seq_dim)
        sinusoid = torch.einsum("..., d->...d", position, self.inv_freq)
        sin, cos = torch.sin(sinusoid), torch.cos(sinusoid)
        return sin, cos

class MultiHeadedScaling(nn.Module):
    """
    Perform an element wise scale shift
    """

    def __init__(
            self,
            shape: typing.Union[int, typing.List[int], torch.Size],
            num_heads: int,
            on_out_ready: typing.Optional[
                typing.Callable[[torch.Tensor], torch.Tensor]
            ],
            dtype: typing.Optional[torch.dtype] = None,
    ) -> None:
        """

        Args:
            shape: the shape of the input dimensions
            num_heads: the number of dimensions to squeeze to
            dtype: the dtype of the parameters at generation
            on_out_ready: the function called on exit
        """
        super(MultiHeadedScaling, self).__init__()

        factory_kwargs = nn.factory_kwargs({"dtype": dtype})
        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        shape = list(tuple(shape))
        self.unsqueeze_dim = - (len(shape) + 1)
        
        shape.insert(0, num_heads)
        self.shape = shape
        self.split_dims = [1] * num_heads
        self.weight = nn.Parameter(torch.empty(self.shape, **factory_kwargs))
        self.bias = nn.Parameter(torch.empty(self.shape, **factory_kwargs))
        self.call_on_out_ready = on_out_ready

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> typing.List[torch.Tensor]:
        """
        Element wise multiplication followed by addition

        Args:
            x: the input tensor with the trailing dimensions following
                ~self.shape

        Returns:
            A output tensor of the same shape

        """
        x = x.unsqueeze(self.unsqueeze_dim) * self.weight + self.bias
        positive_index = x.ndim + self.unsqueeze_dim
        if self.call_on_out_ready is not None:
            # lambda x: self.rope(x, x.ndim - 3)
            x = self.call_on_out_ready(x) 

        x = x.split(self.split_dims, dim=positive_index)
        return [x_i.squeeze(positive_index) for x_i in x] 

    def reset_parameters(self):
        nn.init.normal_(self.weight, std=0.02)
        nn.init.zeros_(self.bias)

class GatedAttentionUnit(nn.Module):
    """
    """
    def __init__(self, cfg):
        super(GatedAttentionUnit, self).__init__()
        self.cfg = cfg
        
        self.gva_proj = nn.Sequential( 
            Linear(cfg.MODEL.EMBEDDER.HIDDEN_DIM, cfg.MODEL.EMBEDDER.PROJECT_DIM * 2 + cfg.MODEL.EMBEDDER.ATTN_DIM),
            nn.SiLU()
        )
        self.multi_headed_scaling = MultiHeadedScaling(
            cfg.MODEL.EMBEDDER.ATTN_DIM,
            num_heads=2,
            on_out_ready=lambda x: self.rope(x, x.ndim - 3)
        )
        self.rope = RoPE(cfg.MODEL.EMBEDDER.ATTN_DIM)
        self.relpos = RelPosEmbedder(cfg.MODEL.EMBEDDER.NUM_RELPOS, embedding_dim=1)
        self.output_proj = Linear(cfg.MODEL.EMBEDDER.PROJECT_DIM, cfg.MODEL.EMBEDDER.HIDDEN_DIM)

    def forward(self,node,scaling,bias):
        """
        The forward method of this class

        Args:
            node: the node representation
            scaling: logits scaling
            bias:
        Returns:

        """
        cfg = self.cfg

        gates, values, base = self.gva_proj(node).split(
            [cfg.MODEL.EMBEDDER.PROJECT_DIM, cfg.MODEL.EMBEDDER.PROJECT_DIM, cfg.MODEL.EMBEDDER.ATTN_DIM], dim=-1
        )
        queries, keys = self.multi_headed_scaling(base)
        node, edge = attention(
            query=queries,
            key=keys,
            scale=scaling,
            value=values,
            bias=bias + self.relpos(base.shape[-2])[..., 0],
            subbatch_size=self.cfg.SOLVER.SUBBATCH_SIZE,
            return_edge=True,
            edge_reduction='sum',
            edge_reduction_dim=-3,
        )

        node = node * gates
        node = self.output_proj(node)
        return node, edge


class OmegaPLMLayer(nn.Module):
    """One OmegaPLM Layer

    This layer baked the pre-layernorm configuration into the model

    Attributes:
        gau: the underlying GAU layer containing most of the computations
    """

    def __init__(self, cfg):
        super(OmegaPLMLayer, self).__init__()
        self.gau = GatedAttentionUnit(cfg)

    def forward(self,node,qk_scaling,bias):
        """Forward method for pre-layernorm

        One layer of OmegaPLM

        Args:
            node: the node representation
            qk_scaling:  the scaling of logits before attention
            bias: the bias for logits before attention

        Returns:
            node and edge representation

        """
        shortcut, node = node, _naive_layernorm(node)
        node, edge = self.gau(node, qk_scaling, bias)
        node = node + shortcut
        return node, edge


class EdgeEmbedder(nn.Module):
    """
    Embed the input into node and edge representations

    """

    def __init__(self, cfg):
        super(EdgeEmbedder, self).__init__()

        self.proj_i = nn.Embedding(cfg.MODEL.EMBEDDER.ALPHABET_SIZE, cfg.MODEL.TRUNK.PAIR_DIM)
        self.proj_j = nn.Embedding(cfg.MODEL.EMBEDDER.ALPHABET_SIZE, cfg.MODEL.TRUNK.PAIR_DIM)
        self.relpos = RelPosEmbedder(cfg.MODEL.EMBEDDER.RELPOS_LEN * 2 + 1, cfg.MODEL.TRUNK.PAIR_DIM)

    def forward(self,fasta_sequence,out: torch.Tensor):
        out += self.proj_i(fasta_sequence).unsqueeze(-2)
        out += self.proj_j(fasta_sequence).unsqueeze(-3)
        out += self.relpos(fasta_sequence.size(-1))
        return out


class OFModule(nn.Module):
    """
    The OmegaFold modules
        args: The arguments used for each of the modules
    """

    def __init__(self,cfg):
        super(OFModule, self).__init__()
        self.cfg = cfg

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype


class OmegaPLM(OFModule):
    """Encoder GAU model

    This is the OmegaPLM model in Wu et al. 2022.

    Attributes:
        input_embedding: This is an embedding layer
        layers: the trunk of the network containing modified GAU layers
        output_norm: an output normalization layer

    """

    def __init__(self, cfg):
        super(OmegaPLM, self).__init__(cfg)
        self.input_embedding = nn.Embedding(
            cfg.MODEL.EMBEDDER.MASKED_ALPHABET, cfg.MODEL.EMBEDDER.HIDDEN_DIM, padding_idx=cfg.MODEL.EMBEDDER.PADDING_IDX
        )
        self.layers = nn.ModuleList(
            [OmegaPLMLayer(cfg) for _ in range(cfg.MODEL.EMBEDDER.NUM_LAYERS)]
        )
        self.output_norm = LayerNorm(cfg.MODEL.EMBEDDER.HIDDEN_DIM)
        self.num_pseudo_msa = cfg.MODEL.EMBEDDER.NUM_PSEUDO_MSA

    def forward(self,token_dict):
        """
        Args:
            tokens: A tensor of input tokens,
                of shape [*, seq_len]
            mask: mask indicating the validity of the tokens,
                of shape [*, seq_len]

        Returns: msa_representation, edge_representation

        """
        tokens, p_mask,p_seq_mask = self._aatype2inputs(
            token_dict['aatype'][:,:,0],
            token_dict['seq_mask'][:,:,0],
            self.device, 
            num_pseudo_msa = self.num_pseudo_msa
        )

        qk_scaling = _get_qk_scaling(p_mask.sum(-1), self.cfg.MODEL.EMBEDDER.ATTN_DIM)
        qk_scaling = qk_scaling[..., None, None]
        bias = mask2bias(p_mask[..., None, :])

        node = self.input_embedding(tokens)
        node *= self._get_finetuning_scale(p_mask, tokens)
 
        edges = torch.empty(len(self.layers), node.size()[0],p_mask.shape[-1], p_mask.shape[-1],dtype=node.dtype, device=node.device)
        for i, layer in enumerate(self.layers):
            node, edges[i] = layer(node, qk_scaling, bias)
        node = self.output_norm(node)
        edges /= (p_mask.any(-1).sum() + 1e-5)

        return node, edges, p_mask

    def _get_finetuning_scale(self, mask, tokens) -> torch.Tensor:
        """Token dropout scaling

        This computes the scaling from Rives et al. 2021

        Args:
            mask: the mask indicating the validity of the input sequence

        Returns:

        need to check

        """
        un_masked_ratio_train = 1 - self.cfg.MODEL.EMBEDDER.MASKED_RATIO
        src_lengths = mask.sum(-1)
        mask_ratio_observed = tokens.eq(21).sum(-1).float() / src_lengths
        mask_ratio_observed = torch.where(
            mask_ratio_observed == 1.,
            torch.full_like(mask_ratio_observed, 0.99),
            mask_ratio_observed
        )
        return un_masked_ratio_train / (1 - mask_ratio_observed)[:, None, None]

    def _aatype2inputs(self,batched_aatype,
                        seq_mask,
                        device,
                        num_pseudo_msa = 7,
                        mask_rate=0.14,
                        num_cycle= 1,
                        deterministic=False):
        
        mask = torch.ones_like(batched_aatype).float()
        assert torch.all(batched_aatype.ge(0)), \
            f"Only take 0-20 amino acids as inputs with unknown amino acid " \
            f"indexed as 20"
        batch_size, num_res = batched_aatype.size() # 256
        g = None 
        if deterministic:
            g = torch.Generator()
            g.manual_seed(num_res)

        # pseudo msa is not supported. 
        p_msa = batched_aatype[:, None, :].repeat(1,num_pseudo_msa,1)
        p_msa_seq_mask = seq_mask[:,None,:].repeat(1,num_pseudo_msa+1,1)
        
        p_msa_mask = torch.rand([batch_size,num_pseudo_msa, num_res], generator=g).gt(mask_rate).to(device)
        p_msa_mask = torch.cat((seq_mask.unsqueeze(1), p_msa_mask), dim=1)
        
        p_msa = torch.cat((batched_aatype.unsqueeze(1), p_msa), dim=1)
        msa_masking = p_msa_seq_mask.bool() & ~p_msa_mask.bool()
        p_msa[~p_msa_seq_mask.bool()] = 20
        p_msa[msa_masking.bool()] = 21
        msa_masking = ~msa_masking

        return p_msa, p_msa_seq_mask,  msa_masking.float()


@EMBEDDER_REGISTRY.register()
class OMEGAPLM(EMBEDDER):
    def __init__(self, cfg):
        
        super().__init__()
        weight_key = cfg.MODEL.EMBEDDER.WEIGHT_KEY
        self.model = OmegaPLM(cfg)
        
        if weight_key is not None:
            self.model.load_state_dict(torch.load(weight_key))

        if cfg.MODEL.EMBEDDER.FREEZE:
            self.freeze()
            self.model.eval()

        self.plm_node_embedder = Linear(cfg.MODEL.EMBEDDER.HIDDEN_DIM, cfg.MODEL.TRUNK.MSA_DIM)
        self.plm_edge_embedder = Linear(cfg.MODEL.EMBEDDER.NUM_LAYERS, cfg.MODEL.TRUNK.PAIR_DIM)
        self.input_embedder = EdgeEmbedder(cfg)

    def forward(self, batched_inputs):

        node, edge, mask = self.model(batched_inputs)
        batched_inputs['masking_index'] = torch.cat(batched_inputs["aatype"].shape[-1] * [mask.unsqueeze(-1)], dim=-1)

        # projection
        node_repre = self.plm_node_embedder(_naive_layernorm(node, in_place=True))
        edge_repre = edge.permute(1, 2, 3, 0)
        edge_repre = self.plm_edge_embedder(_naive_layernorm(edge_repre, in_place=True))
        edge_repre = self.input_embedder(batched_inputs['aatype'][..., 0], out=edge_repre)
        batched_inputs['representations'] = node_repre
        batched_inputs['attentions'] = edge_repre
        
        return batched_inputs