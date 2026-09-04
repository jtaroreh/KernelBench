import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, scale_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    # Calculate channel index for bias/scale broadcasting
    # x is (N, C, H, W), flattened to (N*C*H*W)
    # channel index = (offsets // HW) % C
    channel_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + channel_idx, mask=mask)
    scale = tl.load(scale_ptr + channel_idx, mask=mask)
    
    # Fusion: x + bias -> x * scale -> sigmoid
    val = (x + bias) * scale
    out = 1.0 / (1.0 + tl.exp(-val))
    
    tl.store(out_ptr + offsets, out, mask=mask)

def triton_fused_post_conv(x: torch.Tensor, bias: torch.Tensor, scale: torch.Tensor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    fused_post_conv_kernel[grid](
        x, bias, scale, out,
        n_elements, C, HW,
        BLOCK_SIZE=1024
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)

    def forward(self, x):
        x = self.conv(x)
        # Fusing: x + bias, x * scale, and sigmoid into one kernel
        x = triton_fused_post_conv(x, self.bias.view(-1), self.scale.view(-1))
        x = self.group_norm(x)
        return x