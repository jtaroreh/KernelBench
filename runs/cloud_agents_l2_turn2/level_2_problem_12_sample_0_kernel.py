import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_linear_scalar_mul_leaky_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    multiplier, negative_slope,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_bn[:, None] * stride_bk + offs_k[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs + k * stride_ak, mask=(offs_am[:, None] < M) & (k + offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs + k * stride_bk, mask=(offs_bn[:, None] < N) & (k + offs_k[None, :] < K), other=0.0)
        accumulator += tl.dot(a, tl.trans(b))

    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0)
    accumulator += bias[None, :]
    accumulator *= multiplier
    accumulator = tl.where(accumulator >= 0, accumulator, accumulator * negative_slope)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.linear_layer = nn.Linear(in_features, out_features)
        self.multiplier_val = multiplier
        self.negative_slope_val = negative_slope

    def forward(self, x):
        M, K = x.shape
        N = self.linear_layer.out_features
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        # Weight is (N, K), we need to pass it such that the kernel reads it correctly
        # The kernel expects B_ptr to be (N, K) and uses tl.dot(a, tl.trans(b))
        weight = self.linear_layer.weight
        bias = self.linear_layer.bias
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = lambda META: (tl.cdiv(M, META['BLOCK_M']) * tl.cdiv(N, META['BLOCK_N']),)
        
        fused_linear_scalar_mul_leaky_relu_kernel[grid](
            x, weight, bias, out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            out.stride(0), out.stride(1),
            self.multiplier_val, self.negative_slope_val,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=8
        )
        return out