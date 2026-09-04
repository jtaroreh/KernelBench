import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + (offsets % n_cols), mask=mask, other=0.0)
    
    # Bias Add
    x = x + bias
    
    # Hardtanh: clamp(-1, 1)
    x = tl.where(x < -1.0, -1.0, x)
    x = tl.where(x > 1.0, 1.0, x)
    
    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # Use stable softplus: log(1 + exp(x))
    # Clamp x for exp stability
    x_clamped = tl.clamp(x, -88.0, 88.0)
    softplus = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_clamped)))
    x = x * tl.math.tanh(softplus)
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_ops(x: torch.Tensor, bias: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    n_cols = x.shape[-1]
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_activation_kernel[grid](x, bias, out, n_elements, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    """
    A model that performs a GEMM, BiasAdd, Hardtanh, Mish, and GroupNorm operations in sequence.
    """
    def __init__(self, in_features, out_features, bias_shape, num_groups):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.groupnorm = nn.GroupNorm(num_groups=num_groups, num_channels=out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        x = self.gemm(x)
        x = triton_fused_ops(x, self.bias)
        x = self.groupnorm(x)
        return x