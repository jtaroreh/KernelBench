import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mish_add_hardtanh_mul_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in input/output
    add_value_f32,  # Scalar value to add
    scale_f32,  # Scalar value to multiply
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # --- Mish activation ---
    # softplus(x) = log(1 + exp(x))
    softplus_x = tl.log(1 + tl.exp(x))
    # tanh(softplus(x))
    tanh_softplus_x = tl.tanh(softplus_x)
    # Mish(x) = x * tanh(softplus(x))
    mish_x = x * tanh_softplus_x

    # --- Add value ---
    x_after_add = mish_x + add_value_f32

    # --- Hardtanh activation (min_val=-1, max_val=1) ---
    x_after_hardtanh = tl.maximum(tl.minimum(x_after_add, 1.0), -1.0)

    # --- Scale (multiply) ---
    out = x_after_hardtanh * scale_f32

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_activation_ops(x: torch.Tensor, add_value: float, scale: float):
    """
    This function wraps the Triton kernel call for fused Mish, Add, Hardtanh, and Multiply.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    # Ensure the input is contiguous for efficient memory access in the kernel
    x = x.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Tunable parameter for block size. Larger blocks can be more efficient.

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_mish_add_hardtanh_mul_kernel[grid](
        x,
        out,
        n_elements,
        add_value,  # Pass scalar add_value
        scale,      # Pass scalar scale
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a transposed convolution, then a fused
    Mish activation, add, Hardtanh activation, and scaling using a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # Store add_value and scale as floats for direct passing to Triton kernel
        self.add_value = float(add_value)
        self.scale = float(scale)

    def forward(self, x):
        # Step 1: Convolution Transpose (using PyTorch's optimized implementation)
        x = self.conv_transpose(x)

        # Step 2: Fused Mish, Add, Hardtanh, and Multiply using Triton kernel
        x = triton_fused_activation_ops(x, self.add_value, self.scale)
        
        return x

# The following are helper functions for input generation, not part of the model itself.
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 128  
kernel_size  = 3
stride       = 2  
padding      = 1
output_padding = 1
add_value = 0.5
scale = 2

def get_inputs():
    return [torch.rand(batch_size, in_channels, height, width).cuda()] # Ensure inputs are on CUDA

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale]