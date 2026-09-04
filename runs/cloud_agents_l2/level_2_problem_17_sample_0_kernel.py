import torch
import torch.nn as nn
import triton
import triton.language as tl


# Triton Kernels for Fused Instance Normalization and Division

@triton.jit
def instance_norm_mean_var_kernel(
    x_ptr,  # Pointer to the input tensor (B, C, H, W)
    mean_ptr,  # Pointer to the output mean tensor (B, C)
    var_ptr,  # Pointer to the output variance tensor (B, C)
    B, C, H, W,  # Dimensions of the input tensor
    BLOCK_SIZE_HW: tl.constexpr,  # Block size for iterating over H*W elements
):
    """
    Triton kernel to compute mean and variance for each (B, C) slice
    of the input tensor.
    Each program handles one (B, C) slice.
    """
    num_hw_elements = H * W
    program_id_bc = tl.program_id(0)  # Each program handles one (B, C) slice

    # Calculate batch and channel indices for the current program
    b = program_id_bc // C
    c = program_id_bc % C

    # Base pointer for the current (B, C) slice in the input tensor x
    base_x_ptr = x_ptr + b * C * H * W + c * H * W

    # Accumulators for sum and sum of squares for the current slice
    # Initialized to zeros, with a size of BLOCK_SIZE_HW to allow for vectorized accumulation
    sum_acc = tl.zeros((BLOCK_SIZE_HW,), dtype=tl.float32)
    sum_sq_acc = tl.zeros((BLOCK_SIZE_HW,), dtype=tl.float32)

    # Iterate over the H*W dimension in blocks to compute sum and sum_sq
    for i in range(0, num_hw_elements, BLOCK_SIZE_HW):
        # Generate offsets for the current block
        offsets = i + tl.arange(0, BLOCK_SIZE_HW)
        # Create a mask to handle elements near the boundary (H*W not perfectly divisible by BLOCK_SIZE_HW)
        mask = offsets < num_hw_elements
        # Load a block of data from the input tensor
        x_block = tl.load(base_x_ptr + offsets, mask=mask, other=0.0)
        # Accumulate sum and sum of squares
        sum_acc += x_block
        sum_sq_acc += x_block * x_block

    # Reduce the accumulated sums within the program (across BLOCK_SIZE_HW elements)
    total_sum = tl.sum(sum_acc, axis=0)
    total_sum_sq = tl.sum(sum_sq_acc, axis=0)

    # Calculate mean and variance
    mean = total_sum / num_hw_elements
    # Variance is calculated as E[x^2] - (E[x])^2
    var = total_sum_sq / num_hw_elements - mean * mean
    # Ensure variance is non-negative due to potential floating point inaccuracies
    var = tl.maximum(0.0, var)

    # Store the calculated mean and variance
    mean_out_ptr = mean_ptr + program_id_bc
    var_out_ptr = var_ptr + program_id_bc
    tl.store(mean_out_ptr, mean)
    tl.store(var_out_ptr, var)


@triton.jit
def instance_norm_normalize_div_kernel(
    x_ptr,  # Pointer to the input tensor (B, C, H, W)
    out_ptr,  # Pointer to the output tensor (B, C, H, W)
    mean_ptr,  # Pointer to the mean tensor (B, C)
    var_ptr,  # Pointer to the variance tensor (B, C)
    B, C, H, W,  # Dimensions of the input tensor
    eps,  # Epsilon for numerical stability in normalization
    divide_by,  # Constant value to divide the normalized output by
    BLOCK_SIZE_HW: tl.constexpr,  # Block size for iterating over H*W elements
):
    """
    Triton kernel to normalize the input tensor using pre-computed mean and variance,
    and then divide the result by a constant.
    Each program handles a block of H*W elements for a specific (B, C) slice.
    """
    num_hw_elements = H * W

    # Each program is identified by a 2D ID: (program_id_bc, program_id_hw)
    program_id_bc = tl.program_id(0)  # Identifies the (B, C) slice
    program_id_hw = tl.program_id(1)  # Identifies the block within the H*W dimension

    # Calculate batch and channel indices for the current program
    b = program_id_bc // C
    c = program_id_bc % C

    # Load the pre-computed mean and variance for this (B, C) slice
    mean = tl.load(mean_ptr + program_id_bc)
    var = tl.load(var_ptr + program_id_bc)
    # Calculate inverse standard deviation
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Base pointers for the current (B, C) slice in input x and output out
    base_x_ptr = x_ptr + b * C * H * W + c * H * W
    base_out_ptr = out_ptr + b * C * H * W + c * H * W

    # Offsets for loading/storing a block of H*W elements
    offsets = program_id_hw * BLOCK_SIZE_HW + tl.arange(0, BLOCK_SIZE_HW)
    # Mask to handle elements near the boundary
    mask = offsets < num_hw_elements

    # Load a block of input data
    x_block = tl.load(base_x_ptr + offsets, mask=mask, other=0.0)

    # Perform normalization: (x - mean) / std
    normalized_x = (x_block - mean) * inv_std
    # Perform the final division by the constant
    output_val = normalized_x / divide_by

    # Store the result to the output tensor
    tl.store(base_out_ptr + offsets, output_val, mask=mask)


def triton_instance_norm_div(x: torch.Tensor, eps: float, divide_by: float):
    """
    Python wrapper to launch the Triton kernels for fused Instance Normalization
    and element-wise division.
    Assumes affine=False for InstanceNorm (no learnable weight/bias).
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    # Ensure input is contiguous for simpler pointer arithmetic in Triton kernels
    x = x.contiguous()

    B, C, H, W = x.shape
    num_bc_slices = B * C  # Total number of (B, C) slices
    num_hw_elements = H * W  # Number of elements in H*W dimension

    # Temporary tensors to store mean and variance (shape B, C)
    mean_tensor = torch.empty((B, C), dtype=x.dtype, device=x.device)
    var_tensor = torch.empty((B, C), dtype=x.dtype, device=x.device)
    # Output tensor
    out_tensor = torch.empty_like(x)

    # Tunable parameters for block sizes. These can be optimized for specific hardware
    # and input dimensions to achieve best performance.
    BLOCK_SIZE_HW_MEAN_VAR = 1024  # For the mean/var computation pass
    BLOCK_SIZE_HW_NORM_DIV = 1024  # For the normalization and division pass

    # --- Pass 1: Compute Mean and Variance ---
    # Grid for the mean/var kernel: one program per (B, C) slice
    grid_mean_var = (num_bc_slices,)
    instance_norm_mean_var_kernel[grid_mean_var](
        x, mean_tensor, var_tensor, B, C, H, W,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW_MEAN_VAR
    )

    # --- Pass 2: Normalize and Divide ---
    # Grid for the normalize/divide kernel: (num_bc_slices, num_blocks_hw)
    # Each program handles a block of H*W elements for a specific (B, C) slice
    grid_norm_div = (num_bc_slices, (num_hw_elements + BLOCK_SIZE_HW_NORM_DIV - 1) // BLOCK_SIZE_HW_NORM_DIV)
    instance_norm_normalize_div_kernel[grid_norm_div](
        x, out_tensor, mean_tensor, var_tensor, B, C, H, W, eps, divide_by,
        BLOCK_SIZE_HW=BLOCK_SIZE_HW_NORM_DIV
    )

    return out_tensor


class ModelNew(nn.Module):
    """
    Optimized model that replaces nn.InstanceNorm2d and the subsequent
    element-wise division by a constant with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # The original nn.InstanceNorm2d is replaced by our Triton kernel,
        # so we don't instantiate it here.
        # self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by
        # Default epsilon value for nn.InstanceNorm2d is 1e-5
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        # Call our custom Triton-based fused Instance Normalization and division
        x = triton_instance_norm_div(x, self.eps, self.divide_by)
        return x