import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _gelu_kernel(x):
    # Abramowitz and Stegun formula 7.1.26 (max error < 2.5e-7) for exact erf GELU
    z = x * 0.7071067811865475  # x / sqrt(2)
    abs_z = tl.abs(z)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0 / (1.0 + p * abs_z)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    exp_term = tl.exp(tl.clamp(-abs_z * abs_z, -88.0, 88.0))
    erf_approx = 1.0 - poly * exp_term
    erf_val = tl.where(z >= 0.0, erf_approx, -erf_approx)

    return 0.5 * x * (1.0 + erf_val)

@triton.jit
def layer_norm_gelu_scale_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_cols, eps, scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = x_ptr + row_idx * n_cols
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    
    # Load data
    x = tl.load(row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    # LayerNorm stats across the last dimension
    mean = tl.sum(x, axis=0) / n_cols
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / n_cols
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    # Normalize
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    y = (x - mean) * inv_std * w + b
    
    # GELU + Scaling
    y = _gelu_kernel(y) * scaling_factor
    
    tl.store(out_ptr + row_idx * n_cols + cols, y, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = float(scaling_factor)
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        out = torch.empty_like(x)
        
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols
        
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        layer_norm_gelu_scale_kernel[(n_rows,)](
            x, 
            self.layer_norm.weight, 
            self.layer_norm.bias, 
            out,
            n_cols, 
            self.eps, 
            self.scaling_factor,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        return out