# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html

import torch
import triton
import triton.language as tl


def get_configs_linear_io_bound():
    # https://github.com/kakaobrain/trident/blob/main/trident/kernel/linear.py
    configs = []
    for block_size_n in [64, 128]:
        for num_stages in [2, 3]:
            for num_warps in [2, 4]:
                configs.append(triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8},
                                             num_stages=num_stages, num_warps=num_warps))
    return configs


@triton.autotune(
    configs=get_configs_linear_io_bound(),
    key=['M', 'N']
)
@triton.jit
def _matmul_bias_fwd_fused(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Bias
    bias_ptr, stride_bias,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # Add bias
    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_bn * stride_bias
        bias = tl.load(bias_ptrs)
        accumulator += bias[None, :]
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "sigmoid":
        accumulator = tl.sigmoid(accumulator)
    c = accumulator.to(a_ptr.dtype.element_ty)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * \
        offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


class MatmulBiasTritonFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, bias, activation, is_b_shape_kn):
        # Check constraints.
        assert a.is_contiguous(), "Matrix A must be contiguous"
        assert b.is_contiguous(), "Matrix B must be contiguous"
        if is_b_shape_kn:
            assert a.shape[1] == b.shape[0], "Shapes must be A: (M, K), B: (K, N)"
        else:
            assert a.shape[1] == b.shape[1], "Shapes must be A: (M, K), B: (N, K)"
        if bias is not None:
            assert bias.is_contiguous(), "Bias must be contiguous"
            assert bias.ndim == 1, "Bias must be 1D"
            if is_b_shape_kn:
                assert bias.shape[0] == b.shape[1], "Shapes must be B: (K, N), BIAS: (N)"
            else:
                assert bias.shape[0] == b.shape[0], "Shapes must be B: (N, K), BIAS: (N)"

        M, K = a.shape
        if is_b_shape_kn:
            K, N = b.shape
        else:
            N, K = b.shape

        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        # 1D launch kernel where each block gets its own program.
        def grid(META): return (
            triton.cdiv(M, META['BLOCK_SIZE_M']) *
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

        if is_b_shape_kn:
            stride_b_k, stride_b_n = b.stride(0), b.stride(1)
        else:
            stride_b_n, stride_b_k = b.stride(0), b.stride(1)

        if bias is not None:
            stride_bias = bias.stride(0)
        else:
            stride_bias = None

        _matmul_bias_fwd_fused[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            stride_b_k, stride_b_n,
            c.stride(0), c.stride(1),
            bias, stride_bias,
            ACTIVATION=activation
        )

        return c

    @staticmethod
    def backward(ctx, dout):
        raise RuntimeError(
            "MatmulBiasTritonFunc backward() not implemented yet!")