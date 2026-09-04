import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def gemm_bias_relu_kernel(
    A, B, Bias, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A_ptr = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptr = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    
    # Mask for loading/storing out-of-bounds elements
    mask_m = rm < M
    mask_n = rn < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        # Mask for K dimension if K is not a multiple of BLOCK_K (though usually it is, safety first)
        mask_k = rk[None, :] < k_remaining
        
        a = tl.load(A_ptr, mask=mask_m[:, None] & mask_k, other=0.0)
        b = tl.load(B_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        
        acc += tl.dot(a, b)
        
        A_ptr += BLOCK_K * stride_ak
        B_ptr += BLOCK_K * stride_bk

    Bias_ptr = Bias + rn
    bias = tl.load(Bias_ptr, mask=mask_n, other=0.0)
    
    acc += bias[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)

    out_ptrs = Out + (rm[:, None] * stride_om + rn[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(Out.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])

def triton_gemm_bias_relu(x, weight, bias):
    M, K = x.shape
    K_w, N = weight.shape
    assert K == K_w, f"Dimension mismatch: K={K}, K_w={K_w}"
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    gemm_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    return out

class ModelNew(nn.Module):
    """
    Triton-accelerated version of the Model that performs GEMM, Bias Add, and ReLU.
    Matches the exact initialization structure of the reference model.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        return triton_gemm_bias_relu(x, self.gemm.weight.T, self.bias)