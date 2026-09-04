import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def layer_norm_gelu_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, eps, scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row (spatial location)
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * n_cols
    
    # Load row
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    x = tl.load(row_ptr + cols, mask=mask, other=0.0)
    
    # LayerNorm: Mean and Variance
    mean = tl.sum(x, axis=0) / n_cols
    var = tl.sum((x - mean) * (x - mean), axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)
    x_norm = (x - mean) * inv_std * w + b
    
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.tanh(0.7978845608 * (x_norm + 0.044715 * x_norm * x_norm * x_norm)))
    out = x_norm * cdf
    
    # Scale and Store
    tl.store(out_ptr + row_idx * n_cols + cols, out * scaling_factor, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.ln_weight = nn.Parameter(torch.ones(out_channels))
        self.ln_bias = nn.Parameter(torch.zeros(out_channels))
        self.eps = eps
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        
        # Reshape to (N*D*H*W, C) for LayerNorm
        N, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
        out = torch.empty_like(x)
        
        n_rows = x.shape[0]
        n_cols = C
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        grid = (n_rows,)
        layer_norm_gelu_scale_kernel[grid](
            x, self.ln_weight, self.ln_bias, out,
            n_cols, self.eps, self.scaling_factor,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=4
        )
        
        return out.view(N, D, H, W, C).permute(0, 4, 1, 2, 3)