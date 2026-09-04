import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def relu_hardswish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # ReLU: max(0, x)
    x = tl.where(x > 0.0, x, 0.0)
    
    # HardSwish: x * clamp((x + 3) / 6, 0, 1)
    # Since x is already ReLU'd (x >= 0), (x+3)/6 is always >= 0.5.
    # We only need to clamp the upper bound at 1.
    hswish = x * tl.clamp((x + 3.0) / 6.0, 0.0, 1.0)
    
    tl.store(out_ptr + offsets, hswish, mask=mask)

def fused_activation(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    relu_hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fusing ReLU and HardSwish into a single kernel to reduce memory bandwidth
        return fused_activation(x)