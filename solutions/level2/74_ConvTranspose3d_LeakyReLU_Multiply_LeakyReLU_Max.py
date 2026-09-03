import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_mul_kernel(
    x_ptr,
    m_ptr,
    out_ptr,
    n_elements,
    stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # LeakyReLU 1: x = x > 0 ? x : x * 0.2
    x = tl.where(x > 0, x, x * 0.2)
    
    # Multiply by multiplier (broadcasted across C)
    c_idx = (offsets // stride_c) % 32 # 32 is out_channels
    m = tl.load(m_ptr + c_idx)
    x = x * m
    
    # LeakyReLU 2
    x = tl.where(x > 0, x, x * 0.2)
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_ops(x: torch.Tensor, multiplier: torch.Tensor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Calculate stride for channel dimension (N, C, D, H, W)
    stride_c = x.stride(1)
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_activation_mul_kernel[grid](
        x, multiplier, out, n_elements, stride_c, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape).squeeze())
        self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fusing LeakyReLU -> Mul -> LeakyReLU into one kernel
        x = triton_fused_ops(x, self.multiplier)
        x = self.max_pool(x)
        return x