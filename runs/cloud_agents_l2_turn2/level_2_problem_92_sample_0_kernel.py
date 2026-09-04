import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_norm_act_res_lse_kernel(
    x_conv_ptr, x_norm_ptr, out_ptr,
    num_groups, num_channels, spatial_size, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load x_conv (residual)
    x_conv = tl.load(x_conv_ptr + pid)
    
    # Load x_norm (pre-computed GroupNorm output)
    x_norm = tl.load(x_norm_ptr + pid)
    
    # Tanh (using tl.math.tanh)
    x_tanh = tl.math.tanh(x_norm)
    
    # HardSwish: x * relu6(x + 3) / 6
    x_hs = x_tanh * tl.minimum(tl.maximum(x_tanh + 3.0, 0.0), 6.0) / 6.0
    
    # Residual Add
    res = x_conv + x_hs
    
    # Store intermediate for LogSumExp reduction
    tl.store(out_ptr + pid, res)

@triton.jit
def lse_reduce_kernel(
    res_ptr, out_ptr,
    num_channels, spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    max_val = -float('inf')
    for i in range(num_channels):
        val = tl.load(res_ptr + pid + i * spatial_size)
        max_val = tl.maximum(max_val, val)
    
    sum_exp = 0.0
    for i in range(num_channels):
        val = tl.load(res_ptr + pid + i * spatial_size)
        sum_exp += tl.exp(tl.clamp(val - max_val, -88.0, 88.0))
    
    res = max_val + tl.log(sum_exp)
    tl.store(out_ptr + pid, res)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.groups = groups
        self.out_channels = out_channels

    def forward(self, x):
        x_conv = self.conv(x)
        x_norm = self.group_norm(x_conv)
        
        B, C, H, W = x_conv.shape
        spatial = H * W
        
        # Fused kernel for Tanh, HardSwish, and Residual
        res = torch.empty_like(x_conv)
        grid = (B * C * spatial,)
        fused_norm_act_res_lse_kernel[grid](
            x_conv, x_norm, res,
            self.groups, C, spatial, 1e-5, BLOCK_SIZE=1
        )
        
        # LogSumExp reduction
        out = torch.empty(B, 1, H, W, device=x.device)
        grid_lse = (B * spatial,)
        lse_reduce_kernel[grid_lse](
            res, out, C, spatial, BLOCK_SIZE=1
        )
        
        return out