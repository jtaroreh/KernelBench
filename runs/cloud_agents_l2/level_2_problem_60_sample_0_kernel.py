import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    swish = x * tl.sigmoid(x)
    
    tl.store(out_ptr + offsets, swish, mask=mask)

@triton.jit
def hardswish_kernel(
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
    # relu6(z) = min(max(z, 0), 6)
    hswish = x * tl.clamp(x + 3.0, 0.0, 6.0) / 6.0
    
    tl.store(out_ptr + offsets, hswish, mask=mask)

def triton_swish(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_activation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

def triton_hardswish(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    hardswish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace Swish with Triton kernel
        x = triton_swish(x)
        x = self.group_norm(x)
        # Replace HardSwish with Triton kernel
        x = triton_hardswish(x)
        return x