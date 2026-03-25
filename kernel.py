"""
KernelBench Problem L1_P002: 2_Standard_matrix_multiplication_
Level: 1 | Problem ID: 2
Operations: matmul
Difficulty: medium

Source: ScalingIntelligence/KernelBench
Optimized with AutoKernel (https://github.com/RightNow-AI/autokernel)

The agent optimizes ModelNew to outperform the PyTorch reference (Model).
Edit ModelNew.forward() -- use CUDA C++ via compile_cuda() or Triton @jit.
Run `uv run kernelbench/bench_kb.py` to evaluate correctness + speedup.
"""

KERNELBENCH_PROBLEM = {
    "level": 1,
    "problem_id": 2,
    "name": '2_Standard_matrix_multiplication_',
}

import torch
import torch.nn as nn
import torch.nn.functional as F


# Optional: use AutoKernel's CUDA compilation utility for custom CUDA C++ kernels
# from kernels.cuda._compile import compile_cuda
#
# CUDA_SRC = r"""
# #include <torch/extension.h>
# #include <cuda_runtime.h>
# #include <cuda_fp16.h>
#
# __global__ void my_kernel(const float* input, float* output, int N) {
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (idx < N) output[idx] = input[idx];
# }
#
# torch::Tensor my_op_cuda(torch::Tensor input) {
#     auto output = torch::empty_like(input);
#     int N = input.numel();
#     my_kernel<<<(N+255)/256, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
#     return output;
# }
# """
# _mod = None
# def _get_mod():
#     global _mod
#     if _mod is None:
#         _mod = compile_cuda(CUDA_SRC, "my_op_cuda")
#     return _mod

# ============================================================================
# Reference implementation (DO NOT MODIFY below this line)
# ============================================================================

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        return torch.matmul(A, B)

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2

def get_inputs():
    A = torch.rand(M, K)
    B = torch.rand(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed

# ============================================================================
# Optimized implementation (EDIT THIS)
# ============================================================================

# ModelNew must produce outputs matching Model within atol=1e-2, rtol=1e-2.
# Start by copying Model's logic, then optimize with CUDA C++ or Triton.

import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M_size, N_size, K_size,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M_size, BLOCK_M)
    num_pid_n = tl.cdiv(N_size, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K_size, BLOCK_K)):
        k_mask_a = (offs_k[None, :] + k * BLOCK_K) < K_size
        k_mask_b = (offs_k[:, None] + k * BLOCK_K) < K_size
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M_size) & k_mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=k_mask_b & (offs_n[None, :] < N_size), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(C_ptr.dtype.element_ty)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M_size) & (offs_n[None, :] < N_size)
    tl.store(c_ptrs, c, mask=c_mask)


class ModelNew(nn.Module):
    """
    Triton tiled matmul optimized for RDNA4 (gfx1201).
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        M_size, K_size = A.shape
        K_size2, N_size = B.shape
        C = torch.empty((M_size, N_size), device=A.device, dtype=A.dtype)
        # RDNA4: wave32, limited shared memory
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (triton.cdiv(M_size, BLOCK_M) * triton.cdiv(N_size, BLOCK_N),)
        matmul_kernel[grid](
            A, B, C,
            M_size, N_size, K_size,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_M=8,
            num_warps=4, num_stages=1,
        )
        return C

M = 1024 * 2
K = 4096 * 2
N = 2048 * 2
