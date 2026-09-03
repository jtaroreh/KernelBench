import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_div_leaky_relu_kernel(
    x_ptr,
    out_ptr,
    divisor,
    negative_slope,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Fused operation: x / divisor
    val = x / divisor
    
    # Fused operation: LeakyReLU
    out = tl.where(val > 0, val, val * negative_slope)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_div_leaky_relu(x: torch.Tensor, divisor: float, negative_slope: float = 0.01):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_div_leaky_relu_kernel[grid](
        x, out, divisor, negative_slope, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor

    def forward(self, x):
        # Keep convolution as is (highly optimized by cuDNN)
        x = self.conv(x)
        # Replace element-wise division and activation with fused Triton kernel
        return triton_div_leaky_relu(x, self.divisor, negative_slope=0.01)