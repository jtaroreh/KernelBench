import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_add_hardswish_kernel(
    x_ptr, add_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    add_val = tl.load(add_ptr + offsets, mask=mask)
    
    res = x + add_val
    hs = res * tl.clamp(res + 3.0, 0.0, 6.0) / 6.0
    out = res * hs
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_add_hardswish(x: torch.Tensor, add_input: torch.Tensor):
    n_elements = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_add_hardswish_kernel[grid](
        x, add_input, out,
        n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape) -> None:
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, 
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        return triton_add_hardswish(x, add_input)