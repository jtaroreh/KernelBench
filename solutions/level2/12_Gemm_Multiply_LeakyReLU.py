import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_mul_leaky_relu_kernel(
    x_ptr, out_ptr,
    n_elements,
    multiplier, negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask).to(tl.float32)
    x = x * multiplier
    out = tl.where(x >= 0, x, x * negative_slope)
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_fused_mul_leaky_relu(x: torch.Tensor, multiplier: float, negative_slope: float):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_mul_leaky_relu_kernel[grid](
        x, out,
        n_elements,
        float(multiplier), float(negative_slope),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.negative_slope = negative_slope

    def forward(self, x):
        x = self.gemm(x)
        return triton_fused_mul_leaky_relu(x, self.multiplier, self.negative_slope)