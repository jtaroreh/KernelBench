import torch
import torch.nn as nn
import triton
import triton.language as tl

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
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Swish: x * sigmoid(x)
    x = x * tl.sigmoid(x)
    
    # Divide by 2.0
    x = x / 2.0
    
    # Clamp between -1.0 and 1.0
    x = tl.maximum(tl.minimum(x, 1.0), -1.0)
    
    # Tanh using identity: 2.0 * sigmoid(2.0 * x) - 1.0
    # Note: tl.sigmoid is numerically stable in Triton
    x = 2.0 * tl.sigmoid(2.0 * x) - 1.0
    
    # Clamp between -1.0 and 1.0
    x = tl.maximum(tl.minimum(x, 1.0), -1.0)
    
    # Store result
    tl.store(out_ptr + offsets, x, mask=mask)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        # Perform GEMM using PyTorch
        x = self.gemm(x)
        
        # Prepare output buffer
        out = torch.empty_like(x)
        n_elements = x.numel()
        
        # Launch Triton kernel
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_activation_kernel[grid](
            x,
            out,
            n_elements,
            BLOCK_SIZE=1024,
        )
        
        return out