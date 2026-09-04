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
    
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # Note: tl.math.tanh is the correct API
    cdf = 0.5 * (1.0 + tl.math.tanh(0.7978845608 * (lse + 0.044715 * lse * lse * lse)))
    gelu_out = lse * cdf
    
    tl.store(out_ptr + pid, gelu_out)

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        # Reference logic:
        # 1. Gemm
        # 2. Subtract
        # 3. GlobalAvgPool (dim=1)
        # 4. LogSumExp (dim=1)
        # 5. GELU
        # 6. ResidualAdd (x + original_x)
        
        original_x = x.clone()
        x = self.gemm(x)
        
        batch_size = x.shape[0]
        out_features = x.shape[1]
        
        reduced_out = torch.empty((batch_size, 1), device=x.device, dtype=x.dtype)
        
        # Kernel execution
        block_size = triton.next_power_of_2(out_features)
        grid = (batch_size,)
        fused_post_gemm_kernel[grid](
            x, self.subtract, reduced_out,
            batch_size, out_features,
            BLOCK_SIZE_N=block_size
        )
        
        # The reference model performs:
        # x = mean(x, dim=1) -> logsumexp(x, dim=1) -> gelu(x) -> + original_x
        # Since the input to the residual add is (B, 1) and original_x is (B, in_features),
        # the reference implementation implies broadcasting.
        return reduced_out + original_x.mean(dim=1, keepdim=True)