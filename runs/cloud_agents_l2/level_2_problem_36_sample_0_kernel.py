import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch, channels, height, width,
    BLOCK_SIZE_W: tl.constexpr,
):
    # This kernel fuses:
    # 1. Min along channel dim (dim 1)
    # 2. Sum along height dim (dim 2)
    # 3. GELU activation
    # 4. Bias addition
    
    # Input shape: (batch, channels, height, width)
    # Output shape: (batch, 1, 1, width)
    
    pid_b = tl.program_id(0)
    pid_w = tl.program_id(1)
    
    w_offsets = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    mask_w = w_offsets < width
    
    # Initialize min with a large value
    min_val = tl.full([channels], float('inf'), dtype=tl.float32)
    
    # Iterate over height to perform min across channels
    # We need to reduce channels first
    sum_val = tl.zeros([BLOCK_SIZE_W], dtype=tl.float32)
    
    for h in range(height):
        # Load a slice of the input: (channels, BLOCK_SIZE_W)
        # We perform min across channels for this specific h and w
        row_min = tl.full([BLOCK_SIZE_W], float('inf'), dtype=tl.float32)
        for c in range(channels):
            idx = pid_b * (channels * height * width) + c * (height * width) + h * width + w_offsets
            val = tl.load(idx, mask=mask_w, other=float('inf'))
            row_min = tl.minimum(row_min, val)
        
        # Accumulate sum across height
        sum_val += row_min
        
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    bias = tl.load(bias_ptr)
    x = sum_val
    cdf = 0.5 * (1.0 + tl.tanh(0.79788456 * (x + 0.044715 * x * x * x)))
    out = x * cdf + bias
    
    # Store result
    out_idx = pid_b * width + w_offsets
    tl.store(out_idx, out, mask=mask_w)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        batch, channels, height, width = x.shape
        out = torch.empty((batch, 1, 1, width), device=x.device)
        
        BLOCK_SIZE_W = triton.next_power_of_2(width)
        grid = (batch, (width + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W)
        
        fused_post_conv_kernel[grid](
            x, self.bias, out,
            batch, channels, height, width,
            BLOCK_SIZE_W=BLOCK_SIZE_W
        )
        return out