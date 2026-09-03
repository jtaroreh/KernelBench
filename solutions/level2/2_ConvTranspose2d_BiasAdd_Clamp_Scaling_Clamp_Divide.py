import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_clamp_scale_clamp_div_kernel(
    x_ptr,  # Pointer to input tensor (output of conv_transpose)
    bias_ptr,  # Pointer to bias tensor (C, 1, 1)
    out_ptr,  # Pointer to output tensor
    N_ELEMENTS,  # Total number of elements in input/output
    C_DIM,  # Number of channels (for bias broadcasting)
    H_OUT_DIM,  # Height of the output feature map
    W_OUT_DIM,  # Width of the output feature map
    SCALING_FACTOR: tl.constexpr, # The scalar scaling factor
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < N_ELEMENTS

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Calculate channel index for bias broadcasting
    # x is (B, C, H_OUT, W_OUT)
    # bias is (C, 1, 1)
    # The linear index 'offsets' maps to (b, c, h, w)
    # To get 'c' from 'offsets':
    # c_idx = (offsets % (C_DIM * H_OUT_DIM * W_OUT_DIM)) // (H_OUT_DIM * W_OUT_DIM)
    c_idx = (offsets % (C_DIM * H_OUT_DIM * W_OUT_DIM)) // (H_OUT_DIM * W_OUT_DIM)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0) # Bias is (C,1,1), so we just need the channel index

    # Perform fused operations
    # 1. Add bias
    out = x + bias
    # 2. Clamp [0.0, 1.0]
    out = tl.clamp(out, 0.0, 1.0)
    # 3. Multiply by scaling_factor
    out = out * SCALING_FACTOR
    # 4. Clamp [0.0, 1.0] again
    out = tl.clamp(out, 0.0, 1.0)
    # 5. Divide by scaling_factor
    out = out / SCALING_FACTOR

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_op(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    """
    This function wraps the Triton kernel call for fused operations:
    bias_add -> clamp -> scale -> clamp -> div
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Tensors must be FP32."

    # Ensure inputs are contiguous for simpler linear indexing in Triton
    x = x.contiguous()
    bias = bias.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Get dimensions for kernel
    N_ELEMENTS = x.numel()
    # x shape is (B, C, H_OUT, W_OUT)
    # bias shape is (C, 1, 1)
    C_DIM = x.shape[1]
    H_OUT_DIM = x.shape[2]
    W_OUT_DIM = x.shape[3]

    # Tunable parameter for block size
    BLOCK_SIZE = 1024

    # Determine the number of blocks needed
    grid = lambda meta: ((N_ELEMENTS + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_bias_clamp_scale_clamp_div_kernel[grid](
        x,
        bias,
        out,
        N_ELEMENTS,
        C_DIM,
        H_OUT_DIM,
        W_OUT_DIM,
        SCALING_FACTOR=scaling_factor,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model with Triton kernels for fused element-wise operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        # Replace the sequence of element-wise operations with our fused Triton kernel
        x = triton_fused_op(x, self.bias, self.scaling_factor)
        return x