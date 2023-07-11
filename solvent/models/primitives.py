# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# modified from OmegaFold (https://github.com/HeliXonProtein/OmegaFold)
# Copyright 2022 HeliXon Limited

# modified from Modified from OpenFold (https://github.com/aqlaboratory/openfold)
# Copyright 2021 AlQuraishi Laboratory

# modified from Modified from FastFold (https://github.com/hpcaitech/FastFold)
# Copyright 2023 HPC-AI Tech Inc.

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

import importlib
import math
import numbers
from functools import partial
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import truncnorm
from torch.nn import functional as F

from solvent.utils.checkpointing import get_checkpoint_fn
from solvent.utils.chunk_utils import _chunk_slice
from solvent.utils.kernel.cuda_native.attention_core import attention_core
from solvent.utils.kernel.triton.matmul_bias import MatmulBiasTritonFunc
from solvent.utils.kernel.triton.matmul_bias_packed import \
    MatmulBiasPackedTritonFunc
from solvent.utils.tensor_utils import flatten_final_dims, permute_final_dims

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if (deepspeed_is_installed):
    import deepspeed

fa_is_installed = importlib.util.find_spec("flash_attn") is not None
if (fa_is_installed):
    from flash_attn.bert_padding import pad_input, unpad_input
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.flash_attn_interface import \
        flash_attn_unpadded_kvpacked_func

xformers_is_installed = importlib.util.find_spec("xformers") is not None
if (xformers_is_installed):
    import xformers.ops

triton_is_installed = importlib.util.find_spec("triton") is not None
if (triton_is_installed):
    from solvent.utils.kernel.triton.layer_norm import LayerNormTritonFunc


DEFAULT_LMA_Q_CHUNK_SIZE = 1024
DEFAULT_LMA_KV_CHUNK_SIZE = 4096


def _prod(nums):
    out = 1
    for n in nums:
        out = out * n
    return out


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    shape = weights.shape
    f = _calculate_fan(shape, fan)
    scale = scale / max(1, f)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    size = _prod(shape)
    samples = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=size)
    samples = np.reshape(samples, shape)
    with torch.no_grad():
        weights.copy_(torch.tensor(samples, device=weights.device))


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")


def ipa_point_weights_init_(weights):
    with torch.no_grad():
        softplus_inverse_1 = 0.541324854612918
        weights.fill_(softplus_inverse_1)


class Linear(nn.Linear):
    """
    A Linear layer with built-in nonstandard initializations. Called just
    like torch.nn.Linear.

    Implements the initializers in 1.11.4, plus some additional ones found
    in the code.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        init: str = "default",
        init_fn: Optional[Callable[[torch.Tensor, torch.Tensor], None]] = None,
    ):
        """
        Args:
            in_dim:
                The final dimension of inputs to the layer
            out_dim:
                The final dimension of layer outputs
            bias:
                Whether to learn an additive bias. True by default
            init:
                The initializer to use. Choose from:

                "default": LeCun fan-in truncated normal initialization
                "relu": He initialization w/ truncated normal distribution
                "glorot": Fan-average Glorot uniform initialization
                "gating": Weights=0, Bias=1
                "normal": Normal initialization with std=1/sqrt(fan_in)
                "final": Weights=0, Bias=0

                Overridden by init_fn if the latter is not None.
            init_fn:
                A custom initializer taking weight and bias as inputs.
                Overrides init if not None.
        """
        super(Linear, self).__init__(in_dim, out_dim, bias=bias)

        if bias:
            with torch.no_grad():
                self.bias.fill_(0)

        with torch.no_grad():
            if init_fn is not None:
                init_fn(self.weight, self.bias)
            else:
                if init == "default":
                    lecun_normal_init_(self.weight)
                elif init == "relu":
                    he_normal_init_(self.weight)
                elif init == "glorot":
                    glorot_uniform_init_(self.weight)
                elif init == "gating":
                    gating_init_(self.weight)
                    if bias:
                        self.bias.fill_(1.0)
                elif init == "normal":
                    normal_init_(self.weight)
                elif init == "final":
                    final_init_(self.weight)
                else:
                    raise ValueError("Invalid init string.")


class LayerNorm(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNorm, self).__init__()

        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))
        self.bias = nn.Parameter(torch.zeros(c_in))

    def _ori_forward(self, x):
        d = x.dtype
        deepspeed_is_initialized = (
            deepspeed_is_installed and
            deepspeed.utils.is_initialized()
        )
        if (d is torch.bfloat16 and not deepspeed_is_initialized):
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x,
                    self.c_in,
                    self.weight.to(dtype=d),
                    self.bias.to(dtype=d),
                    self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                self.bias,
                self.eps,
            )

        return out

    def _triton_layer_norm(self, x):
        return LayerNormTritonFunc.apply(x, self.c_in, self.weight, self.bias, self.eps)

    def _triton_forward(self, x):
        # https://github.com/hpcaitech/FastFold/blob/0.2.0/fastfold/model/fastnn/kernel/layer_norm.py

        if not triton_is_installed:
            raise ValueError(
                "_triton_forward requires that triton be installed"
            )

        if len(x.shape) >= 3 and x.shape[-3] > 4000:
            out = torch.empty_like(x)
            # set max chunk_size = dim / 2, to max compute efficiency
            chunk_size = min(
                4000 * 4000 // x.shape[-3], (x.shape[-3] + 1) // 2)
            if len(x.shape) == 3:
                for i in range(x.shape[-3]):
                    out[i:i +
                        chunk_size] = self._triton_layer_norm(x[i:i + chunk_size])
            elif len(x.shape) == 4:
                for j in range(x.shape[-4]):
                    for i in range(0, x.shape[-3], chunk_size):
                        out[j, i:i +
                            chunk_size] = self._triton_layer_norm(x[j, i:i + chunk_size])
            else:
                raise RuntimeError("Shape" + x.shape +
                                   "not implemented for layernorm yet!")
            return out
        else:
            return self._triton_layer_norm(x)

    def forward(self, x, use_triton=True):
        if use_triton:
            return self._triton_forward(x)
        else:
            return self._ori_forward(x)


def softmax_no_cast(t: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
        Softmax, but without automatic casting to fp32 when the input is of
        type bfloat16
    """
    d = t.dtype
    deepspeed_is_initialized = (
        deepspeed_is_installed and
        deepspeed.utils.is_initialized()
    )
    if (d is torch.bfloat16 and not deepspeed_is_initialized):
        with torch.cuda.amp.autocast(enabled=False):
            s = torch.nn.functional.softmax(t, dim=dim)
    else:
        s = torch.nn.functional.softmax(t, dim=dim)

    return s


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 0))

    # [*, H, Q, K]
    a = torch.matmul(query, key)

    for b in biases:
        a += b

    a = softmax_no_cast(a, -1)

    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)

    return a


def _attention_chunked_trainable(
    query, key, value, biases, chunk_size, chunk_dim, checkpoint,
):
    if (checkpoint and len(biases) > 2):
        raise ValueError(
            "Checkpointed version permits only permits two bias terms"
        )

    def _checkpointable_attention(q, k, v, b1, b2):
        bs = [b for b in [b1, b2] if b is not None]
        a = _attention(q, k, v, bs)
        return a

    o_chunks = []
    checkpoint_fn = get_checkpoint_fn()
    count = query.shape[chunk_dim]
    for start in range(0, count, chunk_size):
        end = start + chunk_size
        idx = [slice(None)] * len(query.shape)
        idx[chunk_dim] = slice(start, end)
        idx_tup = tuple(idx)
        q_chunk = query[idx_tup]
        k_chunk = key[idx_tup]
        v_chunk = value[idx_tup]

        def _slice_bias(b):
            idx[chunk_dim] = (
                slice(start, end) if b.shape[chunk_dim] != 1 else slice(None)
            )
            return b[tuple(idx)]

        if (checkpoint):
            bias_1_chunk, bias_2_chunk = [
                _slice_bias(b) if b is not None else None
                for b in (biases + [None, None])[:2]
            ]

            o_chunk = checkpoint_fn(_checkpointable_attention,
                                    q_chunk, k_chunk, v_chunk, bias_1_chunk, bias_2_chunk
                                    )
        else:
            bias_chunks = [
                _slice_bias(b) for b in biases
            ]

            o_chunk = _attention(q_chunk, k_chunk, v_chunk, bias_chunks)

        o_chunk = o_chunk.transpose(-2, -3)
        o_chunks.append(o_chunk)

    o = torch.cat(o_chunks, dim=chunk_dim)
    return o


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
        self,
        c_q: int,
        c_k: int,
        c_v: int,
        c_hidden: int,
        no_heads: int,
        gating: bool = True,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(
            self.c_q, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_k = Linear(
            self.c_k, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_v = Linear(
            self.c_v, self.c_hidden * self.no_heads, bias=False, init="glorot"
        )
        self.linear_o = Linear(
            self.c_hidden * self.no_heads, self.c_q, init="final"
        )

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads, init="gating"
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        # [*, H, Q/K, C_hidden]
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _prep_qkv_xformers(self,
                           q_x: torch.Tensor,
                           kv_x: torch.Tensor
                           ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        '''
        xformers.ops.memory_efficient_attention 
        input tensor shape is [*, Q/K, H, C_hidden]
        so doesn't need [*, H, Q/K, C_hidden] transpose
        '''
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)

        return q, k, v

    def _wrap_up(self,
                 o: torch.Tensor,
                 q_x: torch.Tensor
                 ) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        biases: Optional[List[torch.Tensor]] = None,
        use_memory_efficient_kernel: bool = False,
        use_lma: bool = False,
        lma_q_chunk_size: int = DEFAULT_LMA_Q_CHUNK_SIZE,
        lma_kv_chunk_size: int = DEFAULT_LMA_KV_CHUNK_SIZE,
        use_flash: bool = False,
        flash_mask: Optional[torch.Tensor] = None,
        use_xformers: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
            use_memory_efficient_kernel:
                Whether to use a custom memory-efficient attention kernel.
                This should be the default choice for most. If none of the
                "use_<...>" flags are True, a stock PyTorch implementation
                is used instead
            use_lma:
                Whether to use low-memory attention (Staats & Rabe 2021). If
                none of the "use_<...>" flags are True, a stock PyTorch 
                implementation is used instead
            lma_q_chunk_size:
                Query chunk size (for LMA)
            lma_kv_chunk_size:
                Key/Value chunk size (for LMA)
        Returns
            [*, Q, C_q] attention update
        """
        if (use_lma and (lma_q_chunk_size is None or lma_kv_chunk_size is None)):
            raise ValueError(
                "If use_lma is specified, lma_q_chunk_size and "
                "lma_kv_chunk_size must be provided"
            )

        if (use_flash and biases is not None):
            raise ValueError(
                "use_flash is incompatible with the bias option. For masking, "
                "use flash_mask instead"
            )

        attn_options = [use_memory_efficient_kernel, use_lma, use_flash]
        if (sum(attn_options) > 1):
            raise ValueError(
                "Choose at most one alternative attention algorithm"
            )

        if (biases is None):
            biases = []

        # xformers.ops.memory_efficient_attention
        # input tensor shape is [*, Q/K, H, C_hidden]
        # so uses _prep_qkv_xformers()
        if (use_xformers):
            # [*, Q/K, H, C_hidden]
            q, k, v = self._prep_qkv_xformers(q_x, kv_x)
        else:
            # [*, H, Q/K, C_hidden]
            q, k, v = self._prep_qkv(q_x, kv_x)

        # [*, Q, H, C_hidden]
        float16_enabled = (torch.get_autocast_gpu_dtype() == torch.float16)
        if float16_enabled and torch.is_autocast_enabled():
            use_memory_efficient_kernel = False
        if (use_memory_efficient_kernel):
            if (len(biases) > 2):
                raise ValueError(
                    "If use_memory_efficient_kernel is True, you may only "
                    "provide up to two bias terms"
                )
            o = attention_core(q, k, v, *((biases + [None] * 2)[:2]))
            o = o.transpose(-2, -3)
        elif (use_lma):
            biases = [
                b.expand(b.shape[:-2] + (q_x.shape[-2],) + (kv_x.shape[-2],))
                for b in biases
            ]
            o = _lma(q, k, v, biases, lma_q_chunk_size, lma_kv_chunk_size)
            o = o.transpose(-2, -3)
        elif (use_flash):
            o = _flash_attn(q, k, v, flash_mask)
        elif (use_xformers):
            o = _xformers_attn(q, k, v, biases)
        else:
            o = _attention(q, k, v, biases)
            o = o.transpose(-2, -3)

        o = self._wrap_up(o, q_x)

        return o


class GlobalAttention(nn.Module):
    def __init__(self, c_in, c_hidden, no_heads, inf, eps):
        super(GlobalAttention, self).__init__()

        self.c_in = c_in
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.inf = inf
        self.eps = eps

        self.linear_q = Linear(
            c_in, c_hidden * no_heads, bias=False, init="glorot"
        )

        self.linear_k = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_v = Linear(
            c_in, c_hidden, bias=False, init="glorot",
        )
        self.linear_g = Linear(c_in, c_hidden * no_heads, init="gating")
        self.linear_o = Linear(c_hidden * no_heads, c_in, init="final")

        self.sigmoid = nn.Sigmoid()

    def forward(self,
                m: torch.Tensor,
                mask: torch.Tensor,
                use_lma: bool = False,
                ) -> torch.Tensor:
        # [*, N_res, C_in]
        q = torch.sum(m * mask.unsqueeze(-1), dim=-2) / (
            torch.sum(mask, dim=-1)[..., None] + self.eps
        )

        # [*, N_res, H * C_hidden]
        q = self.linear_q(q)
        q *= (self.c_hidden ** (-0.5))

        # [*, N_res, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, C_hidden]
        k = self.linear_k(m)
        v = self.linear_v(m)

        bias = (self.inf * (mask - 1))[..., :, None, :]
        if (not use_lma):
            # [*, N_res, H, N_seq]
            a = torch.matmul(
                q,
                k.transpose(-1, -2),  # [*, N_res, C_hidden, N_seq]
            )
            a += bias
            a = softmax_no_cast(a)

            # [*, N_res, H, C_hidden]
            o = torch.matmul(
                a,
                v,
            )
        else:
            o = _lma(
                q,
                k,
                v,
                [bias],
                DEFAULT_LMA_Q_CHUNK_SIZE,
                DEFAULT_LMA_KV_CHUNK_SIZE
            )

        # [*, N_res, N_seq, C_hidden]
        g = self.sigmoid(self.linear_g(m))

        # [*, N_res, N_seq, H, C_hidden]
        g = g.view(g.shape[:-1] + (self.no_heads, -1))

        # [*, N_res, N_seq, H, C_hidden]
        o = o.unsqueeze(-3) * g

        # [*, N_res, N_seq, H * C_hidden]
        o = o.reshape(o.shape[:-2] + (-1,))

        # [*, N_res, N_seq, C_in]
        m = self.linear_o(o)

        return m


def _lma(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    biases: List[torch.Tensor],
    q_chunk_size: int,
    kv_chunk_size: int,
):
    no_q, no_kv = q.shape[-2], k.shape[-2]

    # [*, H, Q, C_hidden]
    o = q.new_zeros(q.shape)
    for q_s in range(0, no_q, q_chunk_size):
        q_chunk = q[..., q_s: q_s + q_chunk_size, :]
        large_bias_chunks = [
            b[..., q_s: q_s + q_chunk_size, :] for b in biases
        ]

        maxes = []
        weights = []
        values = []
        for kv_s in range(0, no_kv, kv_chunk_size):
            k_chunk = k[..., kv_s: kv_s + kv_chunk_size, :]
            v_chunk = v[..., kv_s: kv_s + kv_chunk_size, :]
            small_bias_chunks = [
                b[..., kv_s: kv_s + kv_chunk_size] for b in large_bias_chunks
            ]

            a = torch.einsum(
                "...hqd,...hkd->...hqk", q_chunk, k_chunk,
            )

            for b in small_bias_chunks:
                a += b

            max_a = torch.max(a, dim=-1, keepdim=True)[0]
            exp_a = torch.exp(a - max_a)
            exp_v = torch.einsum("...hvf,...hqv->...hqf", v_chunk, exp_a)

            maxes.append(max_a.detach().squeeze(-1))
            weights.append(torch.sum(exp_a, dim=-1))
            values.append(exp_v)

        chunk_max = torch.stack(maxes, dim=-3)
        chunk_weights = torch.stack(weights, dim=-3)
        chunk_values = torch.stack(values, dim=-4)

        global_max = torch.max(chunk_max, dim=-3, keepdim=True)[0]
        max_diffs = torch.exp(chunk_max - global_max)
        chunk_values = chunk_values * max_diffs.unsqueeze(-1)
        chunk_weights = chunk_weights * max_diffs

        all_values = torch.sum(chunk_values, dim=-4)
        all_weights = torch.sum(chunk_weights.unsqueeze(-1), dim=-4)

        q_chunk_out = all_values / all_weights

        o[..., q_s: q_s + q_chunk_size, :] = q_chunk_out

    return o


def _flash_attn(q, k, v, kv_mask):
    if (not fa_is_installed):
        raise ValueError(
            "_flash_attn requires that FlashAttention be installed"
        )

    batch_dims = q.shape[:-3]
    no_heads, n, c = q.shape[-3:]
    dtype = q.dtype

    q = q.half()
    k = k.half()
    v = v.half()
    kv_mask = kv_mask.half()

    # [*, B, N, H, C]
    q = q.transpose(-2, -3)
    k = k.transpose(-2, -3)
    v = v.transpose(-2, -3)

    # [B_flat, N, H, C]
    q = q.reshape(-1, *q.shape[-3:])
    k = k.reshape(-1, *k.shape[-3:])
    v = v.reshape(-1, *v.shape[-3:])

    # Flattened batch size
    batch_size = q.shape[0]

    # [B_flat * N, H, C]
    q = q.reshape(-1, *q.shape[-2:])

    q_max_s = n
    q_cu_seqlens = torch.arange(
        0, (batch_size + 1) * n, step=n, dtype=torch.int32, device=q.device
    )

    # [B_flat, N, 2, H, C]
    kv = torch.stack([k, v], dim=-3)
    kv_shape = kv.shape

    # [B_flat, N, 2 * H * C]
    kv = kv.reshape(*kv.shape[:-3], -1)

    kv_unpad, _, kv_cu_seqlens, kv_max_s = unpad_input(kv, kv_mask)
    kv_unpad = kv_unpad.reshape(-1, *kv_shape[-3:])

    out = flash_attn_unpadded_kvpacked_func(
        q,
        kv_unpad,
        q_cu_seqlens,
        kv_cu_seqlens,
        q_max_s,
        kv_max_s,
        dropout_p=0.,
        softmax_scale=1.,  # q has been scaled already
    )

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c)

    out = out.to(dtype=dtype)

    return out


def _xformers_attn(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        biases: List[torch.Tensor]
) -> torch.Tensor:
    if (not xformers_is_installed):
        raise ValueError(
            "_xformers_attn requires that xFormers be installed"
        )

    batch_dims = query.shape[:-3]
    n, no_heads = query.shape[-3:-1]
    c = value.shape[-1]

    # [B_flat, N, H, C]
    query = query.reshape(-1, *query.shape[-3:])
    key = key.reshape(-1, *key.shape[-3:])
    value = value.reshape(-1, *value.shape[-3:])

    # [B_flat, H, Q, K]
    # make memory-contiguous before reducing attention biases
    # more effective since reducing these causes broadcasting
    # so more elements to re-order
    cont_biases = list(map(lambda b: b.contiguous(), biases))
    attn_bias = sum(cont_biases)
    attn_bias = attn_bias.reshape(-1, *attn_bias.shape[-3:])

    out = xformers.ops.memory_efficient_attention(query, key, value, attn_bias.to(query.dtype))

    # [*, B, N, H, C]
    out = out.reshape(*batch_dims, n, no_heads, c)

    return out


def _softmax(
        x: torch.Tensor,
        dim: int,
        *,
        dtype: Optional[torch.dtype] = None,
        in_place: bool = False
) -> torch.Tensor:
    """
    In-place or normal softmax

    Args:
        x: the input tensor
        dim: the dimension along which to perform the softmax
        dtype: the data type
        in_place: if to perform inplace

    Returns:

    """
    if in_place:
        # TODO: CHECK
        x = x.detach() if x.requires_grad else x
        max_val = torch.max(x, dim=dim, keepdim=True)[0].detach()
        # max_val = torch.max(x, dim=dim, keepdim=True)[0]
        torch.sub(x, max_val, out=x)
        torch.exp(x, out=x)
        summed = torch.sum(x, dim=dim, keepdim=True)
        x /= summed
        # TODO: CHECK
        x.requires_grad = True
        return x
    else:
        return torch.softmax(input=x, dim=dim, dtype=dtype)


def _naive_layernorm(
        inputs: torch.Tensor,
        normalized_shape: Optional[
            Union[int, List[int], torch.Size]] = None,
        in_place: bool = False
) -> torch.Tensor:
    """Layer normalization without a module (and weight)

    Args:
        inputs: the input tensor to be normalized
        normalized_shape: the normalized_shape for normalization
        in_place: if to perform the operations in-place

    Returns:
        normalized tensor

    """
    if normalized_shape is None:
        normalized_shape = inputs.shape[-1]
    if isinstance(normalized_shape, numbers.Integral):
        normalized_shape = (normalized_shape,)

    if in_place:
        # This seems to create small discrepancy in result
        dim = list(range(len(inputs.shape))[-len(normalized_shape):])
        inputs -= inputs.mean(dim=dim, keepdim=True)
        inputs *= torch.rsqrt(inputs.var(dim=dim, keepdim=True) + 1e-5)
        return inputs
    else:
        # F.layer_norm seems a bit faster
        return F.layer_norm(inputs, normalized_shape, None, None, 1e-5)


def triton_linear(
        module: nn.Linear,
        input: torch.Tensor,
        activation: str = ""
) -> torch.Tensor:
    assert torch.is_grad_enabled() == False, "triton_linear has no backward"

    weight, bias = module.weight.detach(), module.bias.detach()
    input_2d = input.reshape(-1, input.shape[-1]).detach()

    if torch.is_autocast_enabled():
        autocast_dtype = torch.get_autocast_gpu_dtype()
        weight, bias = weight.to(dtype=autocast_dtype), bias.to(dtype=autocast_dtype)
        input_2d = input_2d.to(dtype=autocast_dtype)

    output_2d = MatmulBiasTritonFunc.apply(
        input_2d, weight, bias, activation, False)

    output = output_2d.reshape(*input.shape[:-1], weight.shape[-1])

    return output


def triton_linear_packed(
        module1: nn.Linear,
        module2: nn.Linear,
        input: torch.Tensor,
        activation: str = ""
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert torch.is_grad_enabled() == False, "triton_linear_packed has no backward"

    weight1, bias1 = module1.weight.detach(), module1.bias.detach()
    weight2, bias2 = module2.weight.detach(), module2.bias.detach()
    input_2d = input.reshape(-1, input.shape[-1]).detach()

    if torch.is_autocast_enabled():
        autocast_dtype = torch.get_autocast_gpu_dtype()
        weight1, bias1 = weight1.to(dtype=autocast_dtype), bias1.to(dtype=autocast_dtype)
        weight2, bias2 = weight2.to(dtype=autocast_dtype), bias2.to(dtype=autocast_dtype)
        input_2d = input_2d.to(dtype=autocast_dtype)

    output1_2d, output2_2d = MatmulBiasPackedTritonFunc.apply(
        input_2d, weight1, weight2, bias1, bias2, activation, False)

    output1 = output1_2d.reshape(*input.shape[:-1], weight1.shape[-1])
    output2 = output2_2d.reshape(*input.shape[:-1], weight2.shape[-1])

    return output1, output2
