import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_clamp_gelu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clamp [-1.0, 1.0]
    x = tl.maximum(tl.minimum(x, 1.0), -1.0)
    
    # Exact GELU matching PyTorch eager: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    cdf = 0.5 * (1.0 + tl.math.erf(x * inv_sqrt2))
    out = x * cdf
    
    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    Optimized with fused Triton kernel for the clamp and GELU epilogue.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
        x = x + self.sum_tensor
        
        x = x.contiguous()
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_clamp_gelu_kernel[grid](
            x, out,
            n_elements, BLOCK_SIZE=BLOCK_SIZE
        )
        return out