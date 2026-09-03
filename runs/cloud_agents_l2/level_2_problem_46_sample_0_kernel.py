import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, out_ptr,
    sub1, sub2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Fused: (x - sub1) -> tanh -> (x - sub2)
    x = x - sub1
    x = tl.tanh(x)
    x = x - sub2
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_ops(x: torch.Tensor, sub1: float, sub2: float):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_post_conv_kernel[grid](x, out, sub1, sub2, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)

    def forward(self, x):
        # Conv is kept as standard PyTorch op for cuDNN optimization
        x = self.conv(x)
        
        # Fused kernel replaces: x - sub1, tanh, x - sub2
        # This reduces memory bandwidth by keeping data in registers/L1
        x = triton_fused_ops(x, self.subtract1_value, self.subtract2_value)
        
        x = self.avgpool(x)
        return x