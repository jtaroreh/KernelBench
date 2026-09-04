import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mish_tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    # We fuse Mish and Tanh: tanh(x * tanh(ln(1 + exp(x))))
    softplus = tl.log(1.0 + tl.exp(x))
    mish = x * tl.tanh(softplus)
    out = tl.tanh(mish)
    
    tl.store(out_ptr + offsets, out, mask=mask)

def fused_mish_tanh(x: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    mish_tanh_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        # Fusing Mish and Tanh into a single kernel to reduce memory bandwidth overhead
        return fused_mish_tanh(x)