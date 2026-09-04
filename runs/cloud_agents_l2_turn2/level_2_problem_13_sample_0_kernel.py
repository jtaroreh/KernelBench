import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def mean_pool_depth_add_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, H, W,
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_bias_c,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Map 4D grid (B, C, H, W) to 3D grid (B, C, H*W)
    pid = tl.program_id(0)
    num_hw = H * W
    idx_b = pid // (C * num_hw)
    idx_c = (pid // num_hw) % C
    idx_hw = pid % num_hw
    idx_h = idx_hw // W
    idx_w = idx_hw % W

    x_slice_ptr = x_ptr + idx_b * stride_x_b + idx_c * stride_x_c + idx_h * stride_x_h + idx_w * stride_x_w
    
    sum_val = 0.0
    for d_offset in range(0, D, BLOCK_SIZE_D):
        d_offsets = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < D
        x_block = tl.load(x_slice_ptr + d_offsets * stride_x_d, mask=mask, other=0.0)
        sum_val += tl.sum(x_block)
    
    mean_val = sum_val / D
    bias_val = tl.load(bias_ptr + idx_c * stride_bias_c)
    
    out_ptr_loc = out_ptr + idx_b * stride_out_b + idx_c * stride_out_c + idx_h * stride_out_h + idx_w * stride_out_w
    tl.store(out_ptr_loc, mean_val + bias_val)

@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    scaling_factor,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    idx_b = pid // (H * W)
    idx_hw = pid % (H * W)
    idx_h = idx_hw // W
    idx_w = idx_hw % W

    x_row_ptr = x_ptr + idx_b * stride_x_b + idx_h * stride_x_h + idx_w * stride_x_w
    out_row_ptr = out_ptr + idx_b * stride_out_b + idx_h * stride_out_h + idx_w * stride_out_w

    # Max for stability
    max_val = -float('inf')
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, tl.max(x_block))
    
    # Sum of exp
    sum_exp = 0.0
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(tl.clamp(x_block - max_val, -88.0, 88.0)))
    
    # Softmax, Tanh, Scale
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=0.0)
        softmax_val = tl.exp(tl.clamp(x_block - max_val, -88.0, 88.0)) / sum_exp
        res = tl.math.tanh(softmax_val) * scaling_factor
        tl.store(out_row_ptr + c_offsets * stride_out_c, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        B, C, D, H, W = x.shape
        out1 = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
        
        grid1 = (B * C * H * W,)
        mean_pool_depth_add_bias_kernel[grid1](
            x, self.bias, out1, B, C, D, H, W,
            *x.stride(), self.bias.stride()[1], *out1.stride(),
            BLOCK_SIZE_D=triton.next_power_of_2(D)
        )
        
        out2 = torch.empty_like(out1)
        grid2 = (B * H * W,)
        softmax_tanh_scale_kernel[grid2](
            out1, out2, B, C, H, W, self.scaling_factor,
            *out1.stride(), *out2.stride(),
            BLOCK_SIZE_C=triton.next_power_of_2(C)
        )
        return out2