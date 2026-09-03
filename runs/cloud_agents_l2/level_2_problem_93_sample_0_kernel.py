import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    out_ptr,
    add_val,
    mul_val,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # x = x + add_value
    x = x + add_val
    
    # x = torch.min(x, 0.0)
    x = tl.minimum(x, 0.0)
    
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Since x <= 0 here, we apply standard GELU
    cdf = 0.5 * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    x = x * cdf
    
    # x = x * multiply_value
    x = x * mul_val
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_post_conv(x: torch.Tensor, add_val: float, mul_val: float):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_post_conv_kernel[grid](x, out, add_val, mul_val, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fuse the elementwise operations: add, min, gelu, mul
        return triton_post_conv(x, self.add_value, self.multiply_value)