import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_dot_product_scale_kernel(
    A_ptr,             # Pointer to input x (M, K)
    B_vec_ptr,         # Pointer to precomputed_vec (K,)
    C_ptr,             # Pointer to output (M, 1)
    M, K,              # M=batch_size, K=input_size
    scalar_factor,     # combined scaling factor (scaling_factor / 2)
    stride_am, stride_ak, # strides for A
    stride_bk,            # stride for B_vec
    stride_cm,            # stride for C
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID for the batch dimension (M)
    m = tl.program_id(0)

    # Accumulator for the dot product for the current row 'm'
    acc = 0.0

    # Iterate over blocks of K
    for k_idx in range(0, K, BLOCK_SIZE_K):
        offsets = k_idx + tl.arange(0, BLOCK_SIZE_K)
        mask = offsets < K

        # Load a block of A (input x) for the current row m
        a_ptrs = A_ptr + m * stride_am + offsets * stride_ak
        a_block = tl.load(a_ptrs, mask=mask, other=0.0)

        # Load a block of B_vec (precomputed_vec)
        b_ptrs = B_vec_ptr + offsets * stride_bk
        b_block = tl.load(b_ptrs, mask=mask, other=0.0)

        # Perform element-wise multiplication and sum reduction
        acc += tl.sum(a_block * b_block)

    # Apply the combined scalar factor
    final_output = acc * scalar_factor

    # Store the result in the output tensor C
    # C_ptr is (M, 1), so we store to the specific row
    c_ptr = C_ptr + m * stride_cm
    tl.store(c_ptr, final_output)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, division, summation, and scaling
    using a custom Triton kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        self.scalar_factor_combined = self.scaling_factor / 2.0
        self.register_buffer('precomputed_vec', None)
        self._update_precomputed_vec()

    def _update_precomputed_vec(self):
        # Computes sum_h(weight.T[k, h]) = sum_h(weight[h, k])
        # This is the sum of each column of the weight matrix
        self.precomputed_vec = torch.sum(self.weight, dim=0).detach().contiguous()

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self._update_precomputed_vec()
        return self

    def forward(self, x: torch.Tensor):
        assert x.is_cuda, "Input tensor must be on CUDA."
        x = x.contiguous()

        if self.precomputed_vec.device != x.device:
            self.precomputed_vec = self.precomputed_vec.to(x.device)

        M, K = x.shape
        out = torch.empty(M, 1, device=x.device, dtype=x.dtype)

        BLOCK_SIZE_K = 256
        grid = (M,)

        fused_dot_product_scale_kernel[grid](
            x,
            self.precomputed_vec,
            out,
            M,
            K,
            self.scalar_factor_combined,
            x.stride(0), x.stride(1),
            self.precomputed_vec.stride(0),
            out.stride(0),
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        return out