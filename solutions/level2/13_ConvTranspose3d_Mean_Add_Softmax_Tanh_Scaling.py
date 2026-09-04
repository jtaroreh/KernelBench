import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, H, W,
    scaling_factor,
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    stride_bias_c,
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # Grid is (B * H * W)
    idx_b = pid // (H * W)
    idx_hw = pid % (H * W)
    idx_h = idx_hw // W
    idx_w = idx_hw % W

    # Pointers to the start of the channel dimension for this (b, h, w)
    x_ptr_base = x_ptr + idx_b * stride_x_b + idx_h * stride_x_h + idx_w * stride_x_w
    out_ptr_base = out_ptr + idx_b * stride_out_b + idx_h * stride_out_h + idx_w * stride_out_w

    # 1. Load and find max for softmax stability
    max_val = -float('inf')
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_vals = tl.load(x_ptr_base + c_offsets * stride_x_c, mask=mask, other=-float('inf'))
        bias_vals = tl.load(bias_ptr + c_offsets * stride_bias_c, mask=mask, other=0.0)
        max_val = tl.maximum(max_val, tl.max(x_vals + bias_vals))

    # 2. Compute sum of exp
    sum_exp = 0.0
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_vals = tl.load(x_ptr_base + c_offsets * stride_x_c, mask=mask, other=0.0)
        bias_vals = tl.load(bias_ptr + c_offsets * stride_bias_c, mask=mask, other=0.0)
        sum_exp += tl.sum(tl.exp(tl.clamp(x_vals + bias_vals - max_val, -88.0, 88.0)))

    # 3. Softmax, Tanh, Scale
    for c_offset in range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_vals = tl.load(x_ptr_base + c_offsets * stride_x_c, mask=mask, other=0.0)
        bias_vals = tl.load(bias_ptr + c_offsets * stride_bias_c, mask=mask, other=0.0)
        
        softmax_val = tl.exp(tl.clamp(x_vals + bias_vals - max_val, -88.0, 88.0)) / sum_exp
        # Tanh via sigmoid identity: 2.0 * sigmoid(2.0 * x) - 1.0
        tanh_val = 2.0 * tl.sigmoid(2.0 * softmax_val) - 1.0
        res = tanh_val * scaling_factor
        tl.store(out_ptr_base + c_offsets * stride_out_c, res, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # 1. ConvTranspose3d
        x = self.conv_transpose(x)
        # 2. Mean pool over depth
        x = x.mean(dim=2) # Result: (B, C, H, W)
        
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        
        # Fused kernel for Bias Add, Softmax, Tanh, Scaling
        grid = (B * H * W,)
        BLOCK_SIZE_C = triton.next_power_of_2(C)
        
        fused_kernel[grid](
            x, self.bias.squeeze(), out,
            B, C, H, W,
            self.scaling_factor,
            *x.stride(), self.bias.stride()[1], *out.stride(),
            BLOCK_SIZE_C=BLOCK_SIZE_C
        )
        
        return out.unsqueeze(2) # Restore depth dim to match original shape (B, C, 1, H, W)