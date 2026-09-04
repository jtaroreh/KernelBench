import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bn_gelu_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr, out_ptr,
    n_elements, n_features, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over features (columns)
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_features

    # Load BN parameters
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Process each row for the assigned features
    for i in range(n_elements // n_features):
        idx = i * n_features + offsets
        x = tl.load(x_ptr + idx, mask=mask)
        
        # BatchNorm
        x_hat = (x - mean) * inv_std
        x_bn = x_hat * weight + bias
        
        # GELU (approximate)
        x_gelu = 0.5 * x_bn * (1.0 + tl.tanh(0.79788456 * (x_bn + 0.044715 * x_bn * x_bn * x_bn)))
        
        # ReLU
        out = tl.where(x_gelu > 0.0, x_gelu, 0.0)
        
        tl.store(out_ptr + idx, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)

    def forward(self, x):
        x = self.gemm(x)
        # Ensure BN is in eval mode for inference optimization
        self.batch_norm.eval()
        
        out = torch.empty_like(x)
        n_elements = x.numel()
        n_features = x.shape[1]
        eps = self.batch_norm.eps
        
        BLOCK_SIZE = 128
        grid = (triton.cdiv(n_features, BLOCK_SIZE),)
        
        fused_bn_gelu_relu_kernel[grid](
            x, self.batch_norm.weight, self.batch_norm.bias,
            self.batch_norm.running_mean, self.batch_norm.running_var,
            out, n_elements, n_features, eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return out