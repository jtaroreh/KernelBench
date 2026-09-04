import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_pool_sum_scale_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    out_features,
    kernel_size,
    scale_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the batch
    pid = tl.program_id(0)
    if pid >= batch_size:
        return

    # The input to this kernel is the output of matmul: (batch_size, out_features)
    # MaxPool1d with kernel_size=2 reduces out_features to out_features // 2
    # Then we sum across the remaining dimension
    
    reduced_dim = out_features // kernel_size
    row_start = pid * out_features
    
    sum_val = 0.0
    
    # Iterate over the reduced dimension
    for i in range(0, reduced_dim, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < reduced_dim
        
        # Load pairs for max pooling
        val1 = tl.load(x_ptr + row_start + offsets * kernel_size, mask=mask, other=-float('inf'))
        val2 = tl.load(x_ptr + row_start + offsets * kernel_size + 1, mask=mask, other=-float('inf'))
        
        # Max pool
        pooled = tl.maximum(val1, val2)
        
        # Sum
        sum_val += tl.sum(pooled, axis=0)
        
    tl.store(out_ptr + pid, sum_val * scale_factor)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor) -> None:
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

    def forward(self, x):
        # Perform matmul using optimized PyTorch implementation
        x = self.matmul(x)
        
        batch_size = x.shape[0]
        out_features = x.shape[1]
        
        # Prepare output
        out = torch.empty(batch_size, device=x.device, dtype=x.dtype)
        
        # Launch fused kernel
        BLOCK_SIZE = 1024
        grid = (batch_size,)
        
        fused_pool_sum_scale_kernel[grid](
            x,
            out,
            batch_size,
            out_features,
            self.kernel_size,
            self.scale_factor,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out