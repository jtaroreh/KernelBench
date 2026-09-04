import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(
    x_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    BLOCK_SIZE_N: tl.constexpr,
):
    m = tl.program_id(0)
    if m >= M:
        return
    
    row_ptr = x_ptr + m * stride_xm
    acc = 0.0
    for n_offset in range(0, N, BLOCK_SIZE_N):
        cols = n_offset + tl.arange(0, BLOCK_SIZE_N)
        mask = cols < N
        vals = tl.load(row_ptr + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals)
        
    tl.store(out_ptr + m, acc)

def triton_row_sum(x: torch.Tensor):
    M, N = x.shape
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)
    BLOCK_SIZE_N = 1024
    grid = (M,)
    row_sum_kernel[grid](
        x, out,
        M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        return triton_row_sum(x)