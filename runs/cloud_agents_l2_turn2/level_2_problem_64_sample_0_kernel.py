import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gelu_op(x):
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # sqrt(2/pi) approx 0.7978845608
    return 0.5 * x * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

@triton.jit
def fused_activation_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # LeakyReLU(0.01) twice
    x = tl.where(x > 0, x, x * 0.01)
    x = tl.where(x > 0, x, x * 0.01)
    
    # GELU twice
    x = gelu_op(x)
    x = gelu_op(x)
    
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_fused_activations(x: torch.Tensor):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    fused_activation_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Gemm
        x = self.linear(x)
        
        # LogSumExp
        x = torch.logsumexp(x, dim=1, keepdim=True)
        
        # Fused LeakyReLU x2 + GELU x2
        # Ensure input is contiguous for Triton kernel
        x = triton_fused_activations(x.contiguous())
        
        return x