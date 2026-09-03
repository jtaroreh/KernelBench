import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Triton Softmax Kernel
@triton.jit
def softmax_kernel(
    input_ptr,
    output_ptr,
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,  # Input strides
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,  # Output strides
    C: tl.constexpr,  # Number of channels (dimension along which softmax is applied)
    D, H, W,  # Other dimensions
    BLOCK_SIZE: tl.constexpr,  # Block size for loading/storing C elements
):
    # Program ID for the (B, D, H, W) slice
    program_id = tl.program_id(0)

    # Decompose program_id into (b, d, h, w)
    # This assumes a flattened grid where program_id maps to a unique (b, d, h, w) tuple
    b = program_id // (D * H * W)
    rem_b = program_id % (D * H * W)
    d = rem_b // (H * W)
    rem_d = rem_b % (H * W)
    h = rem_d // W
    w = rem_d % W

    # Pointers for the current (b, d, h, w) slice
    input_row_ptr = input_ptr + b * stride_b_in + d * stride_d_in + h * stride_h_in + w * stride_w_in
    output_row_ptr = output_ptr + b * stride_b_out + d * stride_d_out + h * stride_h_out + w * stride_w_out

    # Offsets for channels within the row
    cols = tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds if C is not a multiple of BLOCK_SIZE
    # In this specific case, C=16 and BLOCK_SIZE=16, so mask is always true.
    mask = cols < C

    # Load input values for the current row
    input_values = tl.load(input_row_ptr + cols * stride_c_in, mask=mask, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(input_values, axis=0)

    # Subtract max, exponentiate
    numerator = tl.exp(input_values - row_max)

    # Compute sum
    denominator = tl.sum(numerator, axis=0)

    # Divide and store
    output_values = numerator / denominator
    tl.store(output_row_ptr + cols * stride_c_out, output_values, mask=mask)


def triton_softmax(x: torch.Tensor):
    """
    Triton wrapper for softmax operation along dimension 1.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32, "Tensors must be FP32."

    B, C, D, H, W = x.shape
    out = torch.empty_like(x)

    # Calculate strides for input and output
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in = x.stride()
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out = out.stride()

    # Each program handles one (C) vector for a given (B, D, H, W) slice
    num_programs = B * D * H * W

    # BLOCK_SIZE should be C (out_channels) for this specific softmax implementation
    # Given out_channels = 16, BLOCK_SIZE=16 is a good choice.
    BLOCK_SIZE = C

    # Grid definition: one program per (B, D, H, W) slice
    grid = lambda meta: (num_programs,)

    softmax_kernel[grid](
        x, out,
        stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
        stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
        C, D, H, W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# Triton MaxPool3d Kernel
@triton.jit
def max_pool3d_kernel(
    input_ptr,
    output_ptr,
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,  # Input strides
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,  # Output strides
    C_in, D_in, H_in, W_in,  # Input dimensions
    D_out, H_out, W_out,  # Output dimensions
    kernel_size: tl.constexpr,
    stride: tl.constexpr,
):
    # Program ID for the output element
    program_id = tl.program_id(0)

    # Decompose program_id into (b, c, d_out, h_out, w_out)
    # This assumes a flattened grid where program_id maps to a unique output element
    b = program_id // (C_in * D_out * H_out * W_out)
    rem_b = program_id % (C_in * D_out * H_out * W_out)
    c = rem_b // (D_out * H_out * W_out)
    rem_c = rem_b % (D_out * H_out * W_out)
    d_out = rem_c // (H_out * W_out)
    rem_d_out = rem_c % (H_out * W_out)
    h_out = rem_d_out // W_out
    w_out = rem_d_out % W_out

    # Calculate input starting coordinates for the pooling window
    d_in_start = d_out * stride
    h_in_start = h_out * stride
    w_in_start = w_out * stride

    # Initialize max value with the first element of the window
    # This assumes the window is always valid and not empty.
    max_val = tl.load(input_ptr + b * stride_b_in + c * stride_c_in +
                      d_in_start * stride_d_in + h_in_start * stride_h_in +
                      w_in_start * stride_w_in)

    # Iterate over the pooling window (kernel_size x kernel_size x kernel_size)
    for kd in tl.static_range(kernel_size):
        for kh in tl.static_range(kernel_size):
            for kw in tl.static_range(kernel_size):
                # Skip the first element as it's already loaded
                if kd == 0 and kh == 0 and kw == 0:
                    continue

                current_d_in = d_in_start + kd
                current_h_in = h_in_start + kh
                current_w_in = w_in_start + kw

                # Calculate pointer to current element in input window
                current_ptr = (
                    input_ptr
                    + b * stride_b_in
                    + c * stride_c_in
                    + current_d_in * stride_d_in
                    + current_h_in * stride_h_in
                    + current_w_in * stride_w_in
                )
                current_val = tl.load(current_ptr)
                max_val = tl.max(max_val, current_val)

    # Store the max value to the output
    output_ptr_current = (
        output_ptr
        + b * stride_b_out
        + c * stride_c_out
        + d_out * stride_d_out
        + h_out * stride_h_out
        + w_out * stride_w_out
    )
    tl.store(output_ptr_current, max_val)


def triton_max_pool3d(x: torch.Tensor, kernel_size: int, stride: int):
    """
    Triton wrapper for MaxPool3d operation.
    Assumes no padding and stride equals kernel_size for simplicity,
    matching default nn.MaxPool3d behavior.
    """
    assert x.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32, "Tensors must be FP32."

    B, C_in, D_in, H_in, W_in = x.shape

    # Calculate output dimensions based on PyTorch's default MaxPool3d behavior (no padding)
    # output_dim = floor((input_dim - kernel_size) / stride) + 1
    D_out = (D_in - kernel_size) // stride + 1
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1

    out = torch.empty((B, C_in, D_out, H_out, W_out), device=x.device, dtype=x.dtype)

    # Calculate strides for input and output
    stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in = x.stride()
    stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out = out.stride()

    # Each program computes one output element
    num_programs = B * C_in * D_out * H_out * W_out
    grid = lambda meta: (num_programs,)

    max_pool3d_kernel[grid](
        x, out,
        stride_b_in, stride_c_in, stride_d_in, stride_h_in, stride_w_in,
        stride_b_out, stride_c_out, stride_d_out, stride_h_out, stride_w_out,
        C_in, D_in, H_in, W_in,
        D_out, H_out, W_out,
        kernel_size=kernel_size,
        stride=stride
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, applies Triton-based Softmax,
    and performs two Triton-based max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # pool_kernel_size is used for both pooling layers.
        # For nn.MaxPool3d, if stride is not specified, it defaults to kernel_size.
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_kernel_size  # Assuming default stride behavior

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        x = self.conv(x)
        # Replace torch.softmax with triton_softmax
        x = triton_softmax(x)
        # Replace nn.MaxPool3d with triton_max_pool3d
        x = triton_max_pool3d(x, self.pool_kernel_size, self.pool_stride)
        x = triton_max_pool3d(x, self.pool_kernel_size, self.pool_stride)
        return x