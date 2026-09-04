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

    # Fused Activation: x * tanh(softplus(x))
    # softplus(x) = ln(1 + exp(x))
    # tanh(softplus(x)) = tanh(ln(1 + exp(x)))
    # Note: tanh(ln(y)) = (y^2 - 1) / (y^2 + 1) where y = 1 + exp(x)
    y = 1.0 + tl.exp(x)
    y2 = y * y
    act = x * ((y2 - 1.0) / (y2 + 1.0))

    # Batch Norm: (act - mean) / sqrt(var + eps) * weight + bias
    # Assuming channel-wise normalization for 4D tensor (N, C, H, W)
    # This kernel assumes mean/var/weight/bias are broadcasted or handled per-channel
    # For simplicity in this fused implementation, we assume inputs are flattened 
    # and normalization parameters are aligned.
    mean = tl.load(mean_ptr + offsets, mask=mask)
    var = tl.load(var_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)

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
        
        # Prepare BN parameters for the kernel
        # BatchNorm2d stores running_mean/var. We use them for inference-like behavior
        # or current batch stats if training.
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