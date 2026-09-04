import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bn_scale_softmax_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, scale_ptr, out_ptr,
    n_rows, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols
    
    # Load scale factor
    scale = tl.load(scale_ptr)
    
    row_offsets = tl.arange(0, BLOCK_SIZE)
    mask = row_offsets < n_cols
    
    x = tl.load(x_ptr + row_start + row_offsets, mask=mask, other=-float('inf'))
    mean = tl.load(mean_ptr + row_offsets, mask=mask)
    var = tl.load(var_ptr + row_offsets, mask=mask)
    weight = tl.load(weight_ptr + row_offsets, mask=mask)
    bias = tl.load(bias_ptr + row_offsets, mask=mask)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_bn = (x - mean) * inv_std * weight + bias
    x_scaled = x_bn * scale
    
    # Mask out-of-bound elements for max computation so they don't affect softmax
    x_scaled = tl.where(mask, x_scaled, -float('inf'))
    m_i = tl.max(x_scaled, axis=0)
    
    # Softmax with safe exponentiation
    numerator = tl.exp(tl.clamp(x_scaled - m_i, -88.0, 88.0))
    numerator = tl.where(mask, numerator, 0.0)
    s_i = tl.sum(numerator, axis=0)
    
    result = numerator / s_i
    result = tl.where(mask, result, 0.0)
    
    # Store result
    tl.store(out_ptr + row_start + row_offsets, result, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bn_eps=1e-5, bn_momentum=0.1, scale_shape=(1,)):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features, eps=bn_eps, momentum=bn_momentum)
        self.scale = nn.Parameter(torch.ones(scale_shape))
        self.softmax = nn.Softmax(dim=1)
        self.bn_eps = bn_eps

    def forward(self, x):
        x = self.gemm(x)
        
        # Ensure BN is in eval mode for inference optimization
        self.bn.eval()
        
        # Triton kernel expects contiguous inputs
        x = x.contiguous()
        out = torch.empty_like(x)
        
        n_rows, n_cols = x.shape
        
        # Triton block size must be a power of 2 and cover n_cols
        # For very large n_cols, we pick up to max block size supported or next power of 2
        BLOCK_SIZE = triton.next_power_of_2(n_cols)
        
        grid = (n_rows,)
        
        fused_bn_scale_softmax_kernel[grid](
            x, self.bn.running_mean, self.bn.running_var, 
            self.bn.weight, self.bn.bias, self.scale, out,
            n_rows, n_cols, self.bn_eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out