import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_matmul_kernel(
    x_ptr, out_ptr, 
    batch_size, out_features, pool_size, scale_factor,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # This kernel performs: AvgPool1d -> GELU -> Scale -> Max(dim=1)
    # Input x is (batch_size, out_features)
    # AvgPool1d with kernel_size=pool_size reduces out_features to out_features // pool_size
    
    pid = tl.program_id(0)
    num_cols = out_features // pool_size
    
    # Each program handles one row (batch element)
    row_idx = pid
    if row_idx < batch_size:
        row_max = -float('inf')
        
        # Iterate over the pooled segments
        for col_idx in range(num_cols):
            # Compute average of the window
            sum_val = 0.0
            for k in range(pool_size):
                idx = row_idx * out_features + (col_idx * pool_size + k)
                sum_val += tl.load(x_ptr + idx)
            
            avg = sum_val / pool_size
            
            # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            gelu = 0.5 * avg * (1.0 + tl.tanh(0.79788456 * (avg + 0.044715 * avg * avg * avg)))
            
            # Scale
            scaled = gelu * scale_factor
            
            # Max
            if scaled > row_max:
                row_max = scaled
        
        tl.store(out_ptr + row_idx, row_max)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor) -> None:
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        # Matmul remains standard (highly optimized by cuBLAS)
        x = self.matmul(x)
        
        batch_size, out_features = x.shape
        out = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        
        # Launch fused kernel
        grid = (batch_size,)
        fused_post_matmul_kernel[grid](
            x, out,
            batch_size, out_features, self.pool_kernel_size, self.scale_factor,
            BLOCK_SIZE_M=1, BLOCK_SIZE_N=1
        )
        return out