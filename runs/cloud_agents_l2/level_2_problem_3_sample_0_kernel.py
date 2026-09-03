import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton kernel for fused Add + LayerNorm + GELU
@triton.jit
def fused_add_layernorm_gelu_kernel(
    x_ptr,  # Input tensor (B, C, D, H, W)
    sum_weight_ptr, # Scalar sum_weight
    norm_weight_ptr, # LayerNorm weight (W,)
    norm_bias_ptr,   # LayerNorm bias (W,)
    out_ptr, # Output tensor (B, C, D, H, W)
    # Dimensions
    B, C, D, H, W,
    # Strides for x
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    # Strides for out
    stride_out_b, stride_out_c, stride_out_d, stride_out_h, stride_out_w,
    eps, # LayerNorm epsilon
    BLOCK_SIZE_W: tl.constexpr, # Block size for W dimension
):
    """
    Triton kernel to fuse element-wise addition, Layer Normalization, and GELU activation.
    LayerNorm is applied over the last dimension (W).
    Each program instance processes one (B, C, D, H) slice, normalizing across the W dimension.
    """
    # Each program handles one (B, C, D, H) slice, normalizing over W
    # The grid is (B * C * D * H,)
    row_idx = tl.program_id(0)

    # Calculate the B, C, D, H indices from row_idx
    b_idx = row_idx // (C * D * H)
    rem_c_d_h = row_idx % (C * D * H)
    c_idx = rem_c_d_h // (D * H)
    rem_d_h = rem_c_d_h % (D * H)
    d_idx = rem_d_h // H
    h_idx = rem_d_h % H

    # Pointers to the start of the current (b, c, d, h) slice
    x_row_start_ptr = x_ptr + \
                      b_idx * stride_x_b + \
                      c_idx * stride_x_c + \
                      d_idx * stride_x_d + \
                      h_idx * stride_x_h

    out_row_start_ptr = out_ptr + \
                        b_idx * stride_out_b + \
                        c_idx * stride_out_c + \
                        d_idx * stride_out_d + \
                        h_idx * stride_out_h

    # Load sum_weight (scalar)
    sum_val = tl.load(sum_weight_ptr)

    # Load the W elements for the current row
    cols = tl.arange(0, BLOCK_SIZE_W)
    mask = cols < W

    # Load x values, convert to float32
    x = tl.load(x_row_start_ptr + cols * stride_x_w, mask=mask, other=0.0).to(tl.float32)

    # 1. Element-wise addition with sum_weight
    x_added = x + sum_val

    # 2. Layer Normalization
    # Compute mean across the W dimension
    mean = tl.sum(x_added, axis=0) / W
    # Compute variance across the W dimension
    var = tl.sum(tl.pow(x_added - mean, 2), axis=0) / W
    # Normalize
    rstd = 1 / tl.sqrt(var + eps)
    x_normalized = (x_added - mean) * rstd

    # Load LayerNorm weights and biases (which are of shape (W,))
    norm_weight = tl.load(norm_weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    norm_bias = tl.load(norm_bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    # Apply LayerNorm scale and shift
    x_layernormed = x_normalized * norm_weight + norm_bias

    # 3. GELU activation
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    # Constants for GELU
    M_SQRT2PI = 0.7978845608028654 # sqrt(2 / pi)
    M_0_044715 = 0.044715

    cdf = 0.5 * (1.0 + tl.tanh(M_SQRT2PI * (x_layernormed + M_0_044715 * tl.pow(x_layernormed, 3))))
    out = x_layernormed * cdf

    # Store the result
    tl.store(out_row_start_ptr + cols * stride_out_w, out, mask=mask)


def triton_fused_add_layernorm_gelu(
    x: torch.Tensor,
    sum_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    eps: float = 1e-5,
):
    """
    Wrapper function to launch the Triton fused_add_layernorm_gelu_kernel.

    Args:
        x (torch.Tensor): Input tensor from ConvTranspose3d (B, C, D, H, W).
        sum_weight (torch.Tensor): Scalar weight for element-wise addition.
        norm_weight (torch.Tensor): LayerNorm weight tensor (W,).
        norm_bias (torch.Tensor): LayerNorm bias tensor (W,).
        eps (float): Epsilon for LayerNorm.

    Returns:
        torch.Tensor: Output tensor after fused operations.
    """
    assert x.is_cuda and sum_weight.is_cuda and norm_weight.is_cuda and norm_bias.is_cuda, "All tensors must be on CUDA."
    assert x.dtype == torch.float32, "Input tensor must be FP32."

    # Get dimensions
    B, C, D, H, W = x.shape

    # Prepare output tensor
    out = torch.empty_like(x)

    # Calculate grid dimensions: Each program handles one (B, C, D, H) slice.
    num_norm_groups = B * C * D * H
    grid = (num_norm_groups,)

    # Determine BLOCK_SIZE_W. Triton prefers powers of 2.
    # It should be at least W to cover the entire normalization group.
    BLOCK_SIZE_W = triton.next_power_of_2(W)
    if BLOCK_SIZE_W > 2048: # Cap block size to avoid excessive register usage
        BLOCK_SIZE_W = 2048
    if W > 0 and BLOCK_SIZE_W < 64: # Ensure a minimum block size for efficiency if W is not tiny
        BLOCK_SIZE_W = 64
    if W == 0: # Handle edge case of empty W dimension
        BLOCK_SIZE_W = 1 # A block size of 1 is safe for empty dimensions

    # Launch the Triton kernel
    fused_add_layernorm_gelu_kernel[grid](
        x, sum_weight, norm_weight, norm_bias, out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        eps,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D transposed convolution, followed by a fused
    sum+layernorm+GELU, and average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        # Ensure sum_weight is a float32 parameter
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight, dtype=torch.float32))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        # nn.GELU is removed as it's fused into the custom Triton kernel

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace the sequence: x = x + self.sum_weight, x = self.norm(x), x = self.gelu(x)
        # with our custom fused Triton kernel.
        x = triton_fused_add_layernorm_gelu(
            x,
            self.sum_weight,
            self.norm.weight,
            self.norm.bias,
            self.norm.eps # Pass LayerNorm's epsilon
        )
        x = self.avg_pool(x)
        return x