import torch
import torch.nn as nn
import torch.nn.functional as F
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
    BLOCK_SIZE_K: tl.constexpr, # Block size for K dimension
):
    """
    Triton kernel that fuses the following operations:
    1. Matrix multiplication: (x @ weight.T)
    2. Element-wise division: / 2
    3. Row-wise sum: torch.sum(..., dim=1, keepdim=True)
    4. Element-wise scaling: * scaling_factor

    This is achieved by an algorithmic change:
    output[b, 0] = (scaling_factor / 2) * sum_k( x[b, k] * sum_h( weight.T[k, h] ) )

    Let precomputed_vec[k] = sum_h( weight.T[k, h] ).
    Then output[b, 0] = (scaling_factor / 2) * sum_k( x[b, k] * precomputed_vec[k] ).
    This is a dot product between x[b, :] and precomputed_vec[:], followed by a scalar multiplication.

    Each program instance computes one row of the output C.
    """
    # Program ID for the batch dimension (M)
    m = tl.program_id(0)

    # Accumulator for the dot product for the current row 'm'
    acc = tl.zeros([1], dtype=tl.float32)

    # Iterate over blocks of K
    for k_idx in range(0, K, BLOCK_SIZE_K):
        # Load a block of A (input x) for the current row m
        # a_block: (BLOCK_SIZE_K,)
        a_ptrs = A_ptr + m * stride_am + (k_idx + tl.arange(0, BLOCK_SIZE_K)) * stride_ak
        a_block = tl.load(a_ptrs, mask=(k_idx + tl.arange(0, BLOCK_SIZE_K)) < K, other=0.0)

        # Load a block of B_vec (precomputed_vec)
        # b_block: (BLOCK_SIZE_K,)
        b_ptrs = B_vec_ptr + (k_idx + tl.arange(0, BLOCK_SIZE_K)) * stride_bk
        b_block = tl.load(b_ptrs, mask=(k_idx + tl.arange(0, BLOCK_SIZE_K)) < K, other=0.0)

        # Perform element-wise multiplication and sum reduction
        acc += tl.sum(a_block * b_block)

    # Apply the combined scalar factor
    final_output = acc * scalar_factor

    # Store the result in the output tensor C
    c_ptrs = C_ptr + m * stride_cm
    tl.store(c_ptrs, final_output)


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication, division, summation, and scaling
    using a custom Triton kernel.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(ModelNew, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        # Pre-calculate the combined scalar factor
        self.scalar_factor_combined = self.scaling_factor / 2.0

        # Register a buffer for the precomputed vector (sum of weight.T rows)
        # This buffer will be updated when the model is moved to a different device.
        self.register_buffer('precomputed_vec', None)
        self._update_precomputed_vec()

    def _update_precomputed_vec(self):
        """
        Computes `sum_h(self.weight.T[k, h])` for all k and stores it as a buffer.
        This vector is used in the fused Triton kernel.
        It's detached to prevent gradient computation through this precomputation
        if the model is used for inference or if gradients are handled differently.
        """
        # Ensure the computation happens on the same device as self.weight
        self.precomputed_vec = torch.sum(self.weight.T, dim=1).detach().contiguous()

    def to(self, *args, **kwargs):
        """
        Overrides the .to() method to ensure the precomputed_vec buffer
        is updated and moved to the correct device along with other model parameters.
        """
        # Call the parent .to() method first to move self.weight
        super().to(*args, **kwargs)
        # Then update the precomputed_vec on the new device
        self._update_precomputed_vec()
        return self

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Ensure input tensor is on CUDA and contiguous for Triton
        assert x.is_cuda, "Input tensor must be on CUDA."
        x = x.contiguous()

        # Ensure precomputed_vec is on the same device as x
        if self.precomputed_vec.device != x.device:
            self.precomputed_vec = self.precomputed_vec.to(x.device)

        M, K = x.shape  # M=batch_size, K=input_size

        # Prepare the output tensor
        out = torch.empty(M, 1, device=x.device, dtype=x.dtype)

        # Triton kernel tunable parameters
        # BLOCK_SIZE_K determines how many elements of the K dimension are processed
        # in a single block. It should be a power of 2.
        BLOCK_SIZE_K = 128

        # Define the grid for the Triton kernel launch.
        # Each program instance will compute one row of the output (M programs).
        grid = lambda meta: (M,)

        # Launch the fused Triton kernel
        fused_dot_product_scale_kernel[grid](
            x,
            self.precomputed_vec,
            out,
            M,
            K,
            self.scalar_factor_combined,
            x.stride(0), x.stride(1),  # Strides for A (input x)
            self.precomputed_vec.stride(0), # Stride for B_vec (precomputed_vec)
            out.stride(0),             # Stride for C (output)
            BLOCK_SIZE_K=BLOCK_SIZE_K
        )
        return out