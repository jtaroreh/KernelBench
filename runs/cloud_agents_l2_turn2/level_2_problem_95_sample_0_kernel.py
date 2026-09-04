import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_activation_kernel(
    x_ptr,
    add_ptr,
    out_ptr,
    n_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load x and add_value (broadcasted)
    x = tl.load(x_ptr + offsets, mask=mask)
    add_val = tl.load(add_ptr + (offsets % n_cols), mask=mask)
    
    # Add
    x = x + add_val
    
    # Swish: x * sigmoid(x)
    # Clamp exp to avoid overflow
    neg_x = tl.clamp(-x, -88.0, 88.0)
    x = x * (1.0 / (1.0 + tl.exp(neg_x)))
    
    # Tanh (using tl.math.tanh)
    x = tl.math.tanh(x)
    
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # Using the tanh approximation compatible with PyTorch GELU:
    # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x3 = x * x * x
    inner = 0.7978845608 * (x + 0.044715 * x3)
    cdf = 0.5 * (1.0 + tl.math.tanh(inner))
    x = x * cdf
    
    # Hardtanh: clamp(-1, 1)
    x = tl.maximum(-1.0, tl.minimum(1.0, x))
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_activation(x: torch.Tensor, add_value: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    n_cols = x.shape[-1]
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_activation_kernel[grid](
        x, add_value, out, n_elements, n_cols, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape)) 

    def forward(self, x):
        x = self.matmul(x)
        return triton_fused_activation(x, self.add_value)