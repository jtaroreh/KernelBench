import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, out_ptr, 
    subtract_val, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Subtract
    x = x - subtract_val
    
    # HardSwish: x * relu6(x + 3) / 6
    # relu6(x) = min(max(x, 0), 6)
    hswish = x * tl.clamp(x + 3.0, 0.0, 6.0) * (1.0 / 6.0)
    
    # Store
    tl.store(out_ptr + offsets, hswish, mask=mask)

@triton.jit
def fused_mish_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Mish: x * tanh(softplus(x))
    # Stable softplus: ln(1 + exp(x))
    # Clamp x to avoid overflow in exp
    x_clamped = tl.clamp(x, -88.0, 88.0)
    softplus = tl.log(1.0 + tl.exp(x_clamped))
    
    # Tanh identity: tanh(x) = 2 * sigmoid(2 * x) - 1
    tanh_val = 2.0 * tl.sigmoid(2.0 * softplus) - 1.0
    mish = x * tanh_val
    
    tl.store(out_ptr + offsets, mish, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = subtract_value
        self.pool = nn.MaxPool2d(pool_kernel_size)
        self.BLOCK_SIZE = 1024

    def forward(self, x):
        x = self.conv(x)
        
        # Fused Subtract + HardSwish
        n_elements = x.numel()
        out1 = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_post_conv_kernel[grid](x, out1, self.subtract_value, n_elements, BLOCK_SIZE=self.BLOCK_SIZE)
        
        # MaxPool
        x = self.pool(out1)
        
        # Fused Mish
        n_elements_pool = x.numel()
        out2 = torch.empty_like(x)
        grid_pool = lambda meta: (triton.cdiv(n_elements_pool, meta['BLOCK_SIZE']),)
        fused_mish_kernel[grid_pool](x, out2, n_elements_pool, BLOCK_SIZE=self.BLOCK_SIZE)
        
        return out2