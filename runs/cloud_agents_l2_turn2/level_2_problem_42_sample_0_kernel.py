import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, out_ptr,
    batch_size, out_channels, spatial_size,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Fuses:
    1. Global Average Pooling (mean over spatial dims)
    2. Bias Addition
    3. LogSumExp over channels
    4. Sum over spatial (which is now 1x1)
    5. Multiplication by 10.0
    """
    pid = tl.program_id(0)
    
    # Load bias
    bias_offsets = tl.arange(0, BLOCK_SIZE_C)
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_offsets < out_channels, other=0.0)
    
    # Accumulate sum for mean
    sum_val = tl.zeros([BLOCK_SIZE_C], dtype=tl.float32)
    for s in range(spatial_size):
        x_offsets = pid * (out_channels * spatial_size) + bias_offsets * spatial_size + s
        val = tl.load(x_ptr + x_offsets, mask=bias_offsets < out_channels, other=0.0)
        sum_val += val
    
    # Mean + Bias
    mean_val = sum_val / spatial_size
    x = mean_val + bias
    
    # LogSumExp: log(sum(exp(x)))
    # Use max for numerical stability
    max_val = tl.max(x, axis=0)
    exp_sum = tl.sum(tl.exp(x - max_val), axis=0)
    lse = max_val + tl.log(exp_sum)
    
    # Final scale
    tl.store(out_ptr + pid, lse * 10.0)

def triton_fused_post_conv(x: torch.Tensor, bias: torch.Tensor):
    batch_size, out_channels, h, w = x.shape
    spatial_size = h * w
    out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_C = triton.next_power_of_2(out_channels)
    
    grid = (batch_size,)
    fused_post_conv_kernel[grid](
        x, bias, out,
        batch_size, out_channels, spatial_size,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape) -> None:
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Flatten bias for kernel access
        bias_flat = self.bias.view(-1)
        return triton_fused_post_conv(x, bias_flat)