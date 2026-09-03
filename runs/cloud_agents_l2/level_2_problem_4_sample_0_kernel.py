import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def double_mish_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel to compute two consecutive Mish activations in a fused manner.
    Mish(x) = x * tanh(softplus(x))
    softplus(x) = log(1 + exp(x))
    """
    # Calculate the block start offset for the current program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds when loading/storing
    mask = offsets < n_elements

    # Load input values, ensuring FP32 precision
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # First Mish activation: y = x * tanh(softplus(x))
    # Compute exp(x)
    exp_x = tl.exp(x)
    # Compute softplus(x) = log(1 + exp(x))
    softplus_x = tl.log(1.0 + exp_x)
    # Compute tanh(softplus(x))
    tanh_softplus_x = tl.tanh(softplus_x)
    # Compute y = x * tanh(softplus(x))
    y = x * tanh_softplus_x

    # Second Mish activation: z = y * tanh(softplus(y))
    # Compute exp(y)
    exp_y = tl.exp(y)
    # Compute softplus(y) = log(1 + exp(y))
    softplus_y = tl.log(1.0 + exp_y)
    # Compute tanh(softplus(y))
    tanh_softplus_y = tl.tanh(softplus_y)
    # Compute z = y * tanh(softplus(y))
    z = y * tanh_softplus_y

    # Store the final result
    tl.store(out_ptr + offsets, z, mask=mask)


def triton_double_mish(x: torch.Tensor):
    """
    Python wrapper to launch the Triton double_mish_kernel.
    It handles tensor preparation, grid calculation, and kernel launch.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    # Ensure the input tensor is contiguous for efficient memory access in the kernel
    x = x.contiguous()

    # Prepare an empty output tensor with the same shape and dtype as the input
    out = torch.empty_like(x)

    # Total number of elements in the tensor
    n_elements = x.numel()

    # Triton tunable parameter: BLOCK_SIZE determines how many elements each program handles.
    # This value can be tuned for optimal performance on specific hardware.
    BLOCK_SIZE = 1024

    # Define the grid for the kernel launch.
    # For element-wise operations, a 1D grid is typically used, where each block
    # processes a contiguous chunk of elements.
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    double_mish_kernel[grid](x, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    """
    Optimized model that replaces two consecutive Mish activations with a
    single, fused Triton kernel for potential speedups.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ModelNew, self).__init__()
        # The convolution layer is kept as a standard PyTorch operation,
        # as it's highly optimized by cuDNN and complex to re-implement efficiently.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        # Perform the convolution
        x = self.conv(x)
        # Replace the two original torch.nn.functional.mish(x) calls
        # with our custom fused Triton kernel
        x = triton_double_mish(x)
        return x