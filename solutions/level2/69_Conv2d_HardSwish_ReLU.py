import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardswish_relu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # HardSwish: x * relu6(x + 3) / 6
    # relu6(x) = min(max(x, 0), 6)
    hswish = x * tl.clamp(x + 3.0, 0.0, 6.0) / 6.0
    
    # ReLU: max(0, x)
    out = tl.where(hswish > 0.0, hswish, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_hardswish_relu(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    hardswish_relu_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        # Fused HardSwish + ReLU kernel
        return triton_hardswish_relu(x)