import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_scale_bn_kernel(
    x_ptr, scale_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr,
    out_ptr, n_elements, n_features, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate feature index
    feat_idx = (offsets % n_features)
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask)
    scale = tl.load(scale_ptr + feat_idx, mask=mask)
    mean = tl.load(running_mean_ptr + feat_idx, mask=mask)
    var = tl.load(running_var_ptr + feat_idx, mask=mask)
    weight = tl.load(weight_ptr + feat_idx, mask=mask)
    bias = tl.load(bias_ptr + feat_idx, mask=mask)
    
    # Apply scale
    x = x * scale
    
    # Apply BatchNorm (inference mode)
    # Formula: y = (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x = (x - mean) * inv_std * weight + bias
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_scale_bn(x, scale, bn_layer):
    out = torch.empty_like(x)
    n_elements = x.numel()
    n_features = x.shape[-1]
    
    # Ensure tensors are contiguous
    x = x.contiguous()
    
    # Use a block size that is a power of 2 and fits well in memory
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_scale_bn_kernel[grid](
        x, scale, bn_layer.running_mean, bn_layer.running_var, 
        bn_layer.weight, bn_layer.bias, out, 
        n_elements, n_features, bn_layer.eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scale_shape, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.bn = nn.BatchNorm1d(out_features, eps=eps, momentum=momentum)
        # Set to eval mode to ensure running stats are used
        self.bn.eval()

    def forward(self, x):
        # Perform GEMM
        x = self.gemm(x)
        # Fusing the elementwise scale and the BatchNorm inference pass
        # Ensure inputs are float32 for numerical stability
        x = triton_fused_scale_bn(x.float(), self.scale.float(), self.bn)
        return x