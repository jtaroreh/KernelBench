import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_gemm_kernel(
    x_ptr, sub_ptr, out_ptr, 
    batch_size, out_features,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Load row of x and subtract
    row_offsets = pid * out_features + tl.arange(0, BLOCK_SIZE_N)
    mask = tl.arange(0, BLOCK_SIZE_N) < out_features
    
    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0)
    sub = tl.load(sub_ptr + tl.arange(0, BLOCK_SIZE_N), mask=mask, other=0.0)
    
    # Subtract
    val = x - sub
    
    # Mean (GlobalAvgPool)
    sum_val = tl.sum(val, axis=0)
    mean_val = sum_val / out_features
    
    # LogSumExp on a single scalar is just the scalar itself
    lse = mean_val
    
    # GELU via sigmoid identity for tanh: tanh(z) = 2 * sigmoid(2z) - 1
    # 0.7978845608 * (lse + 0.044715 * lse^3)
    arg = 0.7978845608 * (lse + 0.044715 * lse * lse * lse)
    tanh_val = 2.0 * tl.sigmoid(2.0 * arg) - 1.0
    cdf = 0.5 * (1.0 + tanh_val)
    gelu_out = lse * cdf
    
    tl.store(out_ptr + pid, gelu_out)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x.clone()
        x = self.gemm(x)
        
        batch_size = x.shape[0]
        out_features = x.shape[1]
        
        reduced_out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        
        block_size = triton.next_power_of_2(out_features)
        grid = (batch_size,)
        fused_post_gemm_kernel[grid](
            x, self.subtract, reduced_out,
            batch_size, out_features,
            BLOCK_SIZE_N=block_size
        )
        
        return reduced_out + original_x