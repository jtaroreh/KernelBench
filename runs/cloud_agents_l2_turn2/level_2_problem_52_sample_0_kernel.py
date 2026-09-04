import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    n_elements, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)

    # Activation: x * tanh(softplus(x))
    # softplus(x) = ln(1 + exp(x))
    # To avoid overflow, use log(1 + exp(x)) = max(0, x) + log(1 + exp(-abs(x)))
    # However, for tanh(softplus(x)), we can use the identity:
    # tanh(softplus(x)) = tanh(ln(1 + exp(x))) = (exp(x)) / (exp(x) + 2)
    # This is numerically stable for large x.
    
    # Clamp x to prevent exp overflow
    x_clamped = tl.clamp(x, -20.0, 20.0)
    exp_x = tl.exp(x_clamped)
    act = x * (exp_x / (exp_x + 2.0))

    # Load BN parameters
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

    # Batch Norm: (act - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = ((act - mean) * inv_std) * weight + bias

    tl.store(out_ptr + offsets, out, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = self.conv(x)
        
        # Expand parameters to match input shape (N, C, H, W)
        # Triton kernel expects flattened pointers aligned with x
        mean = self.bn.running_mean.view(1, -1, 1, 1).expand_as(x).contiguous()
        var = self.bn.running_var.view(1, -1, 1, 1).expand_as(x).contiguous()
        weight = self.bn.weight.view(1, -1, 1, 1).expand_as(x).contiguous()
        bias = self.bn.bias.view(1, -1, 1, 1).expand_as(x).contiguous()
        
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        fused_activation_bn_kernel[grid](
            x, mean, var, weight, bias, out,
            n_elements, self.eps, BLOCK_SIZE=BLOCK_SIZE
        )
        return out