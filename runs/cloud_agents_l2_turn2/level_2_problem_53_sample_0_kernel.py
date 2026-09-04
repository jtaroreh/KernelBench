import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_gemm_kernel(
    x_ptr,
    out_ptr,
    scaling_factor,
    hardtanh_min,
    hardtanh_max,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Scale
    x = x * scaling_factor
    
    # Hardtanh
    x = tl.maximum(x, hardtanh_min)
    x = tl.minimum(x, hardtanh_max)
    
    # GELU Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Note: tl.math.tanh is the correct way to access tanh in Triton
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
    x = x * cdf
    
    # Store
    tl.store(out_ptr + offsets, x, mask=mask)

def triton_post_gemm(x: torch.Tensor, scaling_factor: float, hardtanh_min: float, hardtanh_max: float):
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    fused_post_gemm_kernel[grid](
        x, out, scaling_factor, hardtanh_min, hardtanh_max, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return out

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, scaling_factor, hardtanh_min, hardtanh_max) -> None:
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        # Perform GEMM using PyTorch's highly optimized cuBLAS backend
        x = self.gemm(x)
        # Fuse scaling, hardtanh, and gelu into a single Triton kernel
        # Ensure input is contiguous for Triton memory access
        return triton_post_gemm(x.contiguous(), self.scaling_factor, self.hardtanh_min, self.hardtanh_max)