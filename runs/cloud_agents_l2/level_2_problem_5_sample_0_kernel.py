import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def subtract_bias_tanh_kernel(
    x_ptr,  # Pointer to input tensor X (output of conv_transpose)
    bias_ptr,  # Pointer to bias tensor (C, 1, 1)
    out_ptr,  # Pointer to output tensor
    N_BATCH,
    N_CHANNELS,
    N_HEIGHT,
    N_WIDTH,
    STRIDE_X_B,  # Stride for batch dimension of X
    STRIDE_X_C,  # Stride for channel dimension of X
    STRIDE_X_H,  # Stride for height dimension of X
    STRIDE_X_W,  # Stride for width dimension of X
    STRIDE_BIAS_C,  # Stride for channel dimension of Bias (bias is C,1,1)
    STRIDE_OUT_B,  # Stride for batch dimension of Out
    STRIDE_OUT_C,  # Stride for channel dimension of Out
    STRIDE_OUT_H,  # Stride for height dimension of Out
    STRIDE_OUT_W,  # Stride for width dimension of Out
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to perform fused (x - bias) and tanh activation.
    The bias tensor is expected to be of shape (C, 1, 1) and will be broadcast
    across batch, height, and width dimensions.
    """
    # Calculate total number of elements
    N_ELEMENTS = N_BATCH * N_CHANNELS * N_HEIGHT * N_WIDTH

    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to ensure we don't go out of bounds
    mask = offsets < N_ELEMENTS

    # Calculate N-D indices (batch, channel, height, width) from flattened offset
    idx_w = offsets % N_WIDTH
    remaining = offsets // N_WIDTH
    idx_h = remaining % N_HEIGHT
    remaining = remaining // N_HEIGHT
    idx_c = remaining % N_CHANNELS
    idx_b = remaining // N_CHANNELS

    # Calculate memory pointers for X, Bias, and Out
    x_ptrs = x_ptr + idx_b * STRIDE_X_B + idx_c * STRIDE_X_C + idx_h * STRIDE_X_H + idx_w * STRIDE_X_W
    # Bias is (C,1,1), so its value only depends on the channel index
    bias_ptrs = bias_ptr + idx_c * STRIDE_BIAS_C
    out_ptrs = out_ptr + idx_b * STRIDE_OUT_B + idx_c * STRIDE_OUT_C + idx_h * STRIDE_OUT_H + idx_w * STRIDE_OUT_W

    # Load values from memory
    x = tl.load(x_ptrs, mask=mask, other=0.0)
    bias_val = tl.load(bias_ptrs, mask=mask, other=0.0)

    # Perform the fused operations: subtract bias, then apply tanh
    subtracted = x - bias_val
    out = tl.tanh(subtracted)

    # Store the result to output memory
    tl.store(out_ptrs, out, mask=mask)


def triton_subtract_bias_tanh(x: torch.Tensor, bias: torch.Tensor):
    """
    Python wrapper for the Triton fused (x - bias) and tanh kernel.
    Ensures inputs are on CUDA and contiguous, then launches the kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."

    # Ensure inputs are contiguous for simpler stride handling in the kernel.
    # nn.ConvTranspose2d typically outputs contiguous tensors, but explicit
    # contiguous calls make the stride calculations predictable.
    x = x.contiguous()
    bias = bias.contiguous()

    # Validate tensor shapes
    assert bias.dim() == 3 and bias.shape[1] == 1 and bias.shape[2] == 1, \
        f"Bias tensor must be (C, 1, 1), but got shape {bias.shape}"
    assert x.dim() == 4, f"Input tensor x must be 4D (B, C, H, W), but got shape {x.shape}"
    assert x.shape[1] == bias.shape[0], \
        f"Channel dimension of x ({x.shape[1]}) must match bias ({bias.shape[0]})"

    # Prepare output tensor with the same shape and dtype as x
    out = torch.empty_like(x)

    # Get dimensions
    N_BATCH, N_CHANNELS, N_HEIGHT, N_WIDTH = x.shape
    N_ELEMENTS = x.numel()

    # Get strides for input, bias, and output tensors
    STRIDE_X_B, STRIDE_X_C, STRIDE_X_H, STRIDE_X_W = x.stride()
    STRIDE_BIAS_C = bias.stride(0)  # Bias is (C,1,1), so stride(0) is for C
    STRIDE_OUT_B, STRIDE_OUT_C, STRIDE_OUT_H, STRIDE_OUT_W = out.stride()

    # Tunable parameter for block size. A common value for element-wise ops.
    BLOCK_SIZE = 1024

    # Determine the number of blocks needed for the grid
    grid = lambda meta: ((N_ELEMENTS + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    subtract_bias_tanh_kernel[grid](
        x, bias, out,
        N_BATCH, N_CHANNELS, N_HEIGHT, N_WIDTH,
        STRIDE_X_B, STRIDE_X_C, STRIDE_X_H, STRIDE_X_W,
        STRIDE_BIAS_C,
        STRIDE_OUT_B, STRIDE_OUT_C, STRIDE_OUT_H, STRIDE_OUT_W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then a fused
    (subtract bias + tanh) operation using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace the sequential x = x - self.bias and x = torch.tanh(x)
        # with our custom fused Triton kernel for potential speedups.
        x = triton_subtract_bias_tanh(x, self.bias)
        return x