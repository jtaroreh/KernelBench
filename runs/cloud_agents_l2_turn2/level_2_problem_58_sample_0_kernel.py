import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * D * H * W)
    
    spatial_idx = offsets
    w = spatial_idx % W
    h = (spatial_idx // W) % H
    d = (spatial_idx // (W * H)) % D
    n = (spatial_idx // (W * H * D))
    
    # Initialize max_val with a very small number
    max_val = tl.full([BLOCK_SIZE], -float('inf'), dtype=tl.float32)
    
    # 1. LogSumExp along dim=1
    # First pass: find max
    for c in range(C):
        idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w
        val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, val)
    
    # Second pass: sum exp
    sum_exp = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for c in range(C):
        idx = n * (C * D * H * W) + c * (D * H * W) + d * (H * W) + h * W + w
        val = tl.load(x_ptr + idx, mask=mask, other=-float('inf'))
        sum_exp += tl.exp(tl.clamp(val - max_val, -88.0, 88.0))
    
    lse = max_val + tl.log(sum_exp)
    
    # 2. HardSwish: x * sigmoid(x + 3) / 6
    # sigmoid(x) = 1 / (1 + exp(-x))
    # Clamp input to exp to avoid overflow
    sig_in = tl.clamp(-(lse + 3.0), -88.0, 88.0)
    hswish = lse * (1.0 / (1.0 + tl.exp(sig_in))) / 6.0
    
    # 3. Subtraction and Clamp
    bias = tl.load(bias_ptr)
    res = hswish - bias
    res = tl.maximum(res, -1.0)
    res = tl.minimum(res, 1.0)
    
    tl.store(out_ptr + offsets, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
        
        n_elements = N * D * H * W
        BLOCK_SIZE = 256
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_post_conv_kernel[grid](
            x, self.bias, out,
            N, C, D, H, W,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out