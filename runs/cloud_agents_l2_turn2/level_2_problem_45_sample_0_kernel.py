import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sigmoid_gemm_kernel(
    A, B, Bias, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    
    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + k * stride_ak)
        # Apply Sigmoid on the fly during load
        a = 1.0 / (1.0 + tl.exp(-a))
        b = tl.load(B_ptr + k * stride_bk)
        acc += tl.dot(a, b)
    
    bias = tl.load(Bias + rn)
    acc += bias[None, :]
    
    tl.store(Out + (rm[:, None] * N + rn[None, :]), acc)

@triton.jit
def logsumexp_kernel(
    X, Out,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ptr = X + pid * N
    
    # Find max for numerical stability
    max_val = -float('inf')
    for i in range(0, N, BLOCK_SIZE):
        vals = tl.load(row_ptr + i + tl.arange(0, BLOCK_SIZE), mask=(i + tl.arange(0, BLOCK_SIZE)) < N, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(vals))
    
    # Sum of exp
    sum_exp = 0.0
    for i in range(0, N, BLOCK_SIZE):
        vals = tl.load(row_ptr + i + tl.arange(0, BLOCK_SIZE), mask=(i + tl.arange(0, BLOCK_SIZE)) < N, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val))
    
    tl.store(Out + pid, max_val + tl.log(sum_exp))

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        # First linear
        x = self.linear1(x)
        
        # Fused Sigmoid + Linear2
        M, K = x.shape
        N = self.output_size
        out_gemm = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
        sigmoid_gemm_kernel[grid](
            x, self.linear2.weight.t(), self.linear2.bias, out_gemm,
            M, N, self.hidden_size,
            x.stride(0), x.stride(1),
            self.linear2.weight.stride(0), self.linear2.weight.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )
        
        # LogSumExp
        out_lse = torch.empty(M, device=x.device, dtype=x.dtype)
        logsumexp_kernel[(M,)](out_gemm, out_lse, M, N, BLOCK_SIZE=128)
        
        return out_lse