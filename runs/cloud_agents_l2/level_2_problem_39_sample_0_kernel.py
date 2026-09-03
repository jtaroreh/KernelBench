import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_bn_kernel(
    x_ptr, scale_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    out_ptr, n_rows, n_cols, eps,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm_offsets = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn_offsets = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load scale and BN parameters
    scale = tl.load(scale_ptr + rn_offsets, mask=rn_offsets < n_cols, other=1.0)
    mean = tl.load(running_mean_ptr + rn_offsets, mask=rn_offsets < n_cols, other=0.0)
    var = tl.load(running_var_ptr + rn_offsets, mask=rn_offsets < n_cols, other=1.0)
    weight = tl.load(weight_ptr + rn_offsets, mask=rn_offsets < n_cols, other=1.0)
    bias = tl.load(bias_ptr + rn_offsets, mask=rn_offsets < n_cols, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    
    for i in range(0, BLOCK_SIZE_M):
        row = rm_offsets[i]
        if row < n_rows:
            # Load input
            x = tl.load(x_ptr + row * n_cols + rn_offsets, mask=rn_offsets < n_cols, other=0.0)
            # Apply scale
            x = x * scale
            # Apply BN: (x - mean) * inv_std * weight + bias
            x = (x - mean) * inv_std * weight + bias
            # Store
            tl.store(out_ptr + row * n_cols + rn_offsets, x, mask=rn_offsets < n_cols)

def triton_fused_scale_bn(x, scale, bn_layer):
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 128
    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M), triton.cdiv(n_cols, BLOCK_SIZE_N))
    
    fused_scale_bn_kernel[grid](
        x, scale, bn_layer.running_mean, bn_layer.running_var, 
        bn_layer.weight, bn_layer.bias, out, 
        n_rows, n_cols, bn_layer.eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)

    def forward(self, x):
        x = self.gemm(x)
        # Ensure BN is in eval mode for inference or handle training state
        if self.training:
            x = x * self.scale
            x = self.bn(x)
        else:
            x = triton_fused_scale_bn(x, self.scale, self.bn)
        return x