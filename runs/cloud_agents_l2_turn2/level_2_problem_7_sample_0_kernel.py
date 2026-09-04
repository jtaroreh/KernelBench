import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Define the Triton kernel for fused activations and bias addition
@triton.jit
def fused_activations_bias_kernel(
    x_ptr,  # Pointer to input tensor (output of conv)
    bias_ptr,  # Pointer to bias tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements in x and out
    channels, # Number of channels (C)
    depth_height_width, # D * H * W (product of depth, height, width)
    negative_slope, # LeakyReLU negative slope
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Calculate channel index for bias loading.
    # The bias is applied per channel.
    # offsets is a global flattened index in (B, C, D, H, W).
    # channel_idx = (offsets // (D * H * W)) % C
    channel_idx = (offsets // depth_height_width) % channels
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    # 1. ReLU: max(x, 0)
    x = tl.maximum(x, 0.0)

    # 2. LeakyReLU: x if x > 0 else x * negative_slope
    x = tl.where(x > 0, x, x * negative_slope)

    # 3. GELU (using erf approximation): 0.5 * x * (1 + erf(x / sqrt(2)))
    # tl.erf expects float32
    x_gelu_arg = x * (1.0 / math.sqrt(2.0)) # Constant 1/sqrt(2)
    x = 0.5 * x * (1.0 + tl.erf(x_gelu_arg))

    # 4. Sigmoid: 1 / (1 + exp(-x))
    x = 1.0 / (1.0 + tl.exp(-x))

    # 5. Add Bias
    out = x + bias

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_activations_bias(x: torch.Tensor, bias: torch.Tensor, negative_slope: float):
    """
    This function wraps the Triton kernel call for fused activations and bias addition.
    It:
      1. Ensures the inputs are contiguous on GPU and are FP32.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Tensors must be FP32."

    # Ensure inputs are contiguous for simpler pointer arithmetic in the kernel.
    # Bias is (C,1,1,1), so contiguous will flatten it to (C,) which is suitable
    # for direct indexing by channel_idx.
    x = x.contiguous()
    bias = bias.contiguous() 

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    
    # Extract dimensions for bias indexing logic within the kernel.
    # x shape is (B, C, D, H, W)
    channels = x.shape[1]
    depth_height_width = x.shape[2] * x.shape[3] * x.shape[4]

    # Tunable parameter for block size. A common block size for element-wise ops
    # is 128, 256, 512, 1024. We choose 1024 for potentially better occupancy.
    BLOCK_SIZE = 1024 

    # Determine the number of blocks needed for the grid.
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_activations_bias_kernel[grid](
        x, bias, out, n_elements, channels, depth_height_width, negative_slope, BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a 3D convolution, followed by a fused Triton kernel
    for ReLU, LeakyReLU, GELU, Sigmoid activations, and bias addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # The 3D convolution is a complex operation, highly optimized in PyTorch/cuDNN.
        # We keep it as a standard nn.Conv3d layer.
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        # The bias is a learnable parameter.
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        # The negative slope for LeakyReLU is a constant.
        self.negative_slope = 0.01 

    def forward(self, x):
        # Perform the 3D convolution
        x = self.conv(x)
        # Replace the sequence of ReLU, LeakyReLU, GELU, Sigmoid, and bias addition
        # with a single fused Triton kernel call for efficiency.
        x = triton_fused_activations_bias(x, self.bias, self.negative_slope)
        return x