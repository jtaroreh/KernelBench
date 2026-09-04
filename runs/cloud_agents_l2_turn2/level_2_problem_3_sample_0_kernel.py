import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_add_layernorm_gelu_kernel(
    x_ptr, sum_weight_ptr, norm_weight_ptr, norm_bias_ptr, out_ptr,
    B, C, D, H, W,
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    eps,
    BLOCK_SIZE_W: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    b_idx = row_idx // (C * D * H)
    rem_c_d_h = row_idx % (C * D * H)
    c_idx = rem_c_d_h // (D * H)
    rem_d_h = rem_c_d_h % (D * H)
    d_idx = rem_d_h // H
    h_idx = rem_d_h % H

    x_row_start_ptr = x_ptr + b_idx * stride_x_b + c_idx * stride_x_c + d_idx * stride_x_d + h_idx * stride_x_h
    out_row_start_ptr = out_ptr + b_idx * stride_out_b + c_idx * stride_out_c + d_idx * stride_out_d + h_idx * stride_out_h

    sum_val = tl.load(sum_weight_ptr)
    cols = tl.arange(0, BLOCK_SIZE_W)
    mask = cols < W

    x = tl.load(x_row_start_ptr + cols * stride_x_w, mask=mask, other=0.0).to(tl.float32)
    x_added = x + sum_val

    mean = tl.sum(x_added, axis=0) / W
    var = tl.sum(tl.pow(x_added - mean, 2), axis=0) / W
    rstd = 1 / tl.sqrt(var + eps)
    x_normalized = (x_added - mean) * rstd

    norm_weight = tl.load(norm_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    norm_bias = tl.load(norm_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    x_layernormed = x_normalized * norm_weight + norm_bias

    M_SQRT2PI = 0.7978845608028654
    M_0_044715 = 0.044715
    cdf = 0.5 * (1.0 + tl.math.tanh(M_SQRT2PI * (x_layernormed + M_0_044715 * tl.pow(x_layernormed, 3))))
    out = x_layernormed * cdf

    tl.store(out_row_start_ptr + cols * stride_out_w, out, mask=mask)

def triton_fused_add_layernorm_gelu(x, sum_weight, norm_weight, norm_bias, eps=1e-5):
    B, C, D, H, W = x.shape
    out = torch.empty_like(x)
    num_norm_groups = B * C * D * H
    BLOCK_SIZE_W = triton.next_power_of_2(W)
    
    fused_add_layernorm_gelu_kernel[(num_norm_groups,)](
        x, sum_weight, norm_weight, norm_bias, out,
        B, C, D, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        eps,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight, dtype=torch.float32))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_fused_add_layernorm_gelu(x, self.sum_weight, self.norm.weight, self.norm.bias, self.norm.eps)
        x = self.avg_pool(x)
        return x