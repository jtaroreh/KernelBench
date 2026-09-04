import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def subtract_bias_tanh_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N_ELEMENTS,
    N_CHANNELS,
    N_HEIGHT,
    N_WIDTH,
    STRIDE_X_B,
    STRIDE_X_C,
    STRIDE_X_H,
    STRIDE_X_W,
    STRIDE_BIAS_C,
    STRIDE_OUT_B,
    STRIDE_OUT_C,
    STRIDE_OUT_H,
    STRIDE_OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    # Calculate indices
    idx_w = offsets % N_WIDTH
    remaining = offsets // N_WIDTH
    idx_h = remaining % N_HEIGHT
    remaining = remaining // N_HEIGHT
    idx_c = remaining % N_CHANNELS
    idx_b = remaining // N_CHANNELS

    # Calculate pointers
    x_ptrs = x_ptr + idx_b * STRIDE_X_B + idx_c * STRIDE_X_C + idx_h * STRIDE_X_H + idx_w * STRIDE_X_W
    bias_ptrs = bias_ptr + idx_c * STRIDE_BIAS_C
    out_ptrs = out_ptr + idx_b * STRIDE_OUT_B + idx_c * STRIDE_OUT_C + idx_h * STRIDE_OUT_H + idx_w * STRIDE_OUT_W

    # Load
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptrs, mask=mask, other=0.0)

    # Compute: x - bias, then tanh using sigmoid identity
    # tanh(x) = 2 * sigmoid(2 * x) - 1
    val = x - bias_val
    out = 2.0 * tl.sigmoid(2.0 * val) - 1.0

    # Store
    tl.store(out_ptrs, out, mask=mask)

def triton_subtract_bias_tanh(x: torch.Tensor, bias: torch.Tensor):
    x = x.contiguous()
    bias = bias.contiguous().view(-1) # Ensure bias is 1D for indexing
    out = torch.empty_like(x)
    
    N_BATCH, N_CHANNELS, N_HEIGHT, N_WIDTH = x.shape
    N_ELEMENTS = x.numel()
    
    STRIDE_X_B, STRIDE_X_C, STRIDE_X_H, STRIDE_X_W = x.stride()
    STRIDE_BIAS_C = bias.stride(0)
    STRIDE_OUT_B, STRIDE_OUT_C, STRIDE_OUT_H, STRIDE_OUT_W = out.stride()
    
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N_ELEMENTS, BLOCK_SIZE),)
    
    subtract_bias_tanh_kernel[grid](
        x, bias, out,
        N_ELEMENTS, N_CHANNELS, N_HEIGHT, N_WIDTH,
        STRIDE_X_B, STRIDE_X_C, STRIDE_X_H, STRIDE_X_W,
        STRIDE_BIAS_C,
        STRIDE_OUT_B, STRIDE_OUT_C, STRIDE_OUT_H, STRIDE_OUT_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_subtract_bias_tanh(x, self.bias)
        return x