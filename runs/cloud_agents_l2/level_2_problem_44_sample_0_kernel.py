import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, out_ptr,
    batch, channels, height, width,
    multiplier,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fuses:
    1. Scalar multiplication
    2. Global Average Pooling (reduction over H and W)
    """
    # Each program handles one channel of one batch item
    pid = tl.program_id(0)
    b = pid // channels
    c = pid % channels
    
    n_elements = height * width
    
    # Pointer to the start of the (b, c) slice
    x_ptr += (b * channels * n_elements) + (c * n_elements)
    
    # Reduction
    sum_val = 0.0
    for i in range(0, n_elements, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(val)
    
    # Apply multiplier and average
    result = (sum_val * multiplier) / n_elements
    
    # Store result (1x1 spatial dimension)
    out_ptr += (b * channels + c)
    tl.store(out_ptr, result)

def triton_fused_post_conv(x: torch.Tensor, multiplier: float):
    batch, channels, height, width = x.shape
    out = torch.empty((batch, channels, 1, 1), device=x.device, dtype=x.dtype)
    
    n_elements = height * width
    grid = (batch * channels,)
    
    fused_post_conv_kernel[grid](
        x, out,
        batch, channels, height, width,
        multiplier,
        BLOCK_SIZE=1024
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.multiplier = multiplier

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fusing multiplication and the two global average pooling operations into one kernel
        # Note: Two sequential global average poolings on the same spatial dims 
        # is mathematically equivalent to one average pooling.
        return triton_fused_post_conv(x, self.multiplier)