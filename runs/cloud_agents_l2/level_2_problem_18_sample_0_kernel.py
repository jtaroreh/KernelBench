import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused Linear + Sum kernel
@triton.jit
def linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,  # M=batch_size, K=in_features, N=out_features
    stride_xm, stride_xk,  # strides for X (M, K)
    stride_wn, stride_wk,  # strides for W (N, K)
    stride_outm,           # stride for output (M, 1)
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    Triton kernel for fused matrix multiplication, bias addition, and row-wise sum.
    Specifically, it computes: output_m = sum_n( (X_m @ W.T)_n + B_n )
    where X is (M, K), W is (N, K), B is (N,), and output is (M, 1).

    Each program instance handles one row of the batch (one 'm' dimension).
    It iterates over blocks of K (in_features) and N (out_features) to perform
    the computation efficiently.
    """
    # Each program computes the sum for one row of the batch
    m = tl.program_id(0)

    # Accumulator for the final sum for this row 'm'
    row_sum_acc = tl.zeros((1,), dtype=tl.float32)

    # Loop over the N dimension (output features) in blocks
    for n_block_start in tl.range(0, N, BLOCK_SIZE_N):
        n_offsets = n_block_start + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < N

        # Accumulator for the current block of N (before summing it up)
        # This will store a part of (X_m @ W.T + B) vector
        partial_row_acc = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        # Loop over the K dimension (in_features) in blocks
        for k_block_start in tl.range(0, K, BLOCK_SIZE_K):
            k_offsets = k_block_start + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < K

            # Load block of X for the current row 'm'
            # x_block shape: (BLOCK_SIZE_K,)
            x_block = tl.load(x_ptr + m * stride_xm + k_offsets * stride_xk,
                              mask=k_mask, other=0.0)

            # Load block of W (weight matrix)
            # W is (N, K). We need to load a (BLOCK_SIZE_N, BLOCK_SIZE_K) block.
            w_block = tl.load(w_ptr + n_offsets[:, None] * stride_wn + k_offsets[None, :] * stride_wk,
                              mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            # Perform dot product: (1, BLOCK_SIZE_K) @ (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # tl.dot(x_block, tl.trans(w_block)) computes (x_block @ w_block.T)
            # x_block is (BLOCK_SIZE_K,), tl.trans(w_block) is (BLOCK_SIZE_K, BLOCK_SIZE_N)
            # Result is (BLOCK_SIZE_N,)
            partial_row_acc += tl.dot(x_block, tl.trans(w_block))

        # Add bias for the current N block
        bias_block = tl.load(b_ptr + n_offsets, mask=n_mask, other=0.0)
        partial_row_acc += bias_block

        # Sum the results for this N block and add to the total row sum
        row_sum_acc += tl.sum(partial_row_acc, axis=0)

    # Store the final sum for this row 'm'
    tl.store(out_ptr + m * stride_outm, row_sum_acc)


def triton_linear_sum(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor):
    """
    Python wrapper for the Triton fused linear + sum kernel.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        weight (torch.Tensor): Weight tensor of shape (out_features, in_features).
        bias (torch.Tensor): Bias tensor of shape (out_features,).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, 1).
    """
    assert x.is_cuda and weight.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and weight.dtype == torch.float32 and bias.dtype == torch.float32, "Tensors must be FP32."

    # Ensure inputs are contiguous for simpler stride calculation.
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    M, K = x.shape
    N, K_w = weight.shape
    assert K == K_w, f"Mismatch in in_features: x has {K}, weight has {K_w}"
    assert bias.shape == (N,), f"Bias shape mismatch: expected ({N},), got {bias.shape}"

    # Output tensor will be (M, 1)
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    # Strides for memory access
    stride_xm, stride_xk = x.stride()
    stride_wn, stride_wk = weight.stride()
    stride_outm = out.stride(0) # For (M, 1) tensor, stride_outm is 1

    # Grid dimensions: one program per batch row (M)
    grid = lambda meta: (M,)

    # Tunable parameters for block sizes. These are chosen as reasonable defaults
    # for the given input dimensions (K=8192, N=8192).
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_N = 128

    linear_sum_kernel[grid](
        x, weight, bias, out,
        M, K, N,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_outm,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a sequence of operations:
        - Fused Matrix multiplication + Bias Add + Sum (using a custom Triton kernel)
        - The subsequent Max, Average pooling, and LogSumExp operations are
          redundant when applied to a (batch_size, 1) tensor with keepdim=True.
          These operations effectively become identity operations and are
          therefore removed for efficiency.
    """
    def __init__(self, in_features, out_features):
        super(ModelNew, self).__init__()
        # We still need the nn.Linear layer to manage its weight and bias parameters
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Use our custom Triton kernel for the fused linear + sum operation.
        # The nn.Linear layer's weight is (out_features, in_features)
        # and its bias is (out_features,).
        return triton_linear_sum(x, self.linear.weight, self.linear.bias)