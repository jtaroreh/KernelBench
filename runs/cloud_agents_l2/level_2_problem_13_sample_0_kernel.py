import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_pool_depth_add_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    D: tl.constexpr,
    # Strides for x (B, C, D, H, W)
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
    # Strides for bias (1, C, 1, 1, 1)
    stride_bias_c,
    # Strides for out (B, C, 1, H, W)
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_D: tl.constexpr,
):
    """
    Triton kernel to perform mean pooling over the depth dimension (D)
    and then add a broadcastable bias.

    Each program instance handles one (batch, channel, height, width) slice.
    It iterates over the depth dimension (D) to compute the mean.
    """
    idx_b = tl.program_id(0)
    idx_c = tl.program_id(1)
    idx_h = tl.program_id(2)
    idx_w = tl.program_id(3)

    # Pointers to the start of the D-dimension for this (b, c, h, w) slice
    x_slice_ptr = x_ptr + idx_b * stride_x_b + idx_c * stride_x_c + idx_h * stride_x_h + idx_w * stride_x_w
    # Bias is (1, C, 1, 1, 1), so we only need to index by channel
    bias_val_ptr = bias_ptr + idx_c * stride_bias_c

    # Compute sum over D
    sum_val = tl.full([BLOCK_SIZE_D], 0.0, dtype=tl.float32)
    for d_offset in tl.static_range(0, D, BLOCK_SIZE_D):
        d_offsets = d_offset + tl.arange(0, BLOCK_SIZE_D)
        mask = d_offsets < D
        x_block = tl.load(x_slice_ptr + d_offsets * stride_x_d, mask=mask, other=0.0)
        sum_val = sum_val + x_block
    sum_val = tl.sum(sum_val, axis=0) # Reduce within the block to get a single sum value

    mean_val = sum_val / D

    # Load bias value
    bias_val = tl.load(bias_val_ptr)

    # Add bias
    result = mean_val + bias_val

    # Store result in the output tensor (depth dimension is 1)
    out_ptr_loc = out_ptr + idx_b * stride_out_b + idx_c * stride_out_c + idx_h * stride_out_h + idx_w * stride_out_w
    tl.store(out_ptr_loc, result)


def triton_mean_pool_add_bias(x: torch.Tensor, bias: torch.Tensor):
    """
    Wrapper function for the mean_pool_depth_add_bias_kernel.
    Performs mean pooling over depth and adds a broadcastable bias.
    Input x: (B, C, D, H, W)
    Input bias: (1, C, 1, 1, 1)
    Output: (B, C, 1, H, W)
    """
    assert x.is_cuda and bias.is_cuda, "Tensors must be on CUDA."
    assert x.dtype == torch.float32 and bias.dtype == torch.float32, "Tensors must be FP32."

    B, C, D, H, W = x.shape

    # Output tensor will have depth dimension 1
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)

    # Get strides for input, bias, and output tensors
    stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w = x.stride()
    # For bias (1, C, 1, 1, 1), we only need the channel stride
    stride_bias_c = bias.stride()[1]
    stride_out_b, stride_out_c, _, stride_out_h, stride_out_w = out.stride()

    # Grid dimensions: (B, C, H, W)
    grid = (B, C, H, W)

    # BLOCK_SIZE_D is a tunable parameter, typically a power of 2
    # We choose the next power of 2 greater than or equal to D
    BLOCK_SIZE_D = triton.next_power_of_2(D) if D > 0 else 1

    mean_pool_depth_add_bias_kernel[grid](
        x, bias, out,
        D,
        stride_x_b, stride_x_c, stride_x_d, stride_x_h, stride_x_w,
        stride_bias_c,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
    )
    return out


@triton.jit
def softmax_tanh_scale_kernel(
    x_ptr, out_ptr,
    C: tl.constexpr,
    scaling_factor,
    # Strides for x (B, C, 1, H, W) - depth stride is not used
    stride_x_b, stride_x_c, stride_x_h, stride_x_w,
    # Strides for out (B, C, 1, H, W) - depth stride is not used
    stride_out_b, stride_out_c, stride_out_h, stride_out_w,
    BLOCK_SIZE_C: tl.constexpr,
):
    """
    Triton kernel to perform softmax over channels (dim=1),
    followed by tanh activation, and then scaling.

    Each program instance handles one (batch, height, width) slice,
    and computes the operations across all channels (C) for that slice.
    """
    idx_b = tl.program_id(0)
    idx_h = tl.program_id(1)
    idx_w = tl.program_id(2)

    # Pointers to the start of the C-dimension for this (b, h, w) slice
    x_row_ptr = x_ptr + idx_b * stride_x_b + idx_h * stride_x_h + idx_w * stride_x_w
    out_row_ptr = out_ptr + idx_b * stride_out_b + idx_h * stride_out_h + idx_w * stride_out_w

    # Phase 1: Compute max for numerical stability (max_val = max(x))
    # Initialize with negative infinity
    max_val = tl.full([BLOCK_SIZE_C], -float('inf'), dtype=tl.float32)
    for c_offset in tl.static_range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=-float('inf'))
        max_val = tl.maximum(max_val, x_block)
    max_val = tl.max(max_val, axis=0) # Reduce within the block to get a single max value

    # Phase 2: Compute sum of exponentials (sum_exp = sum(exp(x - max_val)))
    sum_exp = tl.full([BLOCK_SIZE_C], 0.0, dtype=tl.float32)
    for c_offset in tl.static_range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=0.0)
        exp_x = tl.exp(x_block - max_val) # Subtract max for numerical stability
        sum_exp = sum_exp + exp_x
    sum_exp = tl.sum(sum_exp, axis=0) # Reduce within the block to get a single sum_exp value

    # Phase 3: Compute softmax, tanh, and scaling, then store
    for c_offset in tl.static_range(0, C, BLOCK_SIZE_C):
        c_offsets = c_offset + tl.arange(0, BLOCK_SIZE_C)
        mask = c_offsets < C
        x_block = tl.load(x_row_ptr + c_offsets * stride_x_c, mask=mask, other=0.0)
        exp_x = tl.exp(x_block - max_val)
        softmax_val = exp_x / sum_exp
        tanh_val = tl.tanh(softmax_val)
        scaled_val = tanh_val * scaling_factor
        tl.store(out_row_ptr + c_offsets * stride_out_c, scaled_val, mask=mask)


def triton_softmax_tanh_scale(x: torch.Tensor, scaling_factor: float):
    """
    Wrapper function for the softmax_tanh_scale_kernel.
    Performs softmax over channels, tanh activation, and scaling.
    Input x: (B, C, 1, H, W)
    Output: (B, C, 1, H, W)
    """
    assert x.is_cuda, "Tensor must be on CUDA."
    assert x.dtype == torch.float32, "Tensor must be FP32."

    B, C, _, H, W = x.shape

    out = torch.empty_like(x)

    # Get strides for input and output tensors
    # The depth dimension (index 2) has size 1, so its stride is not used in the kernel
    stride_x_b, stride_x_c, _, stride_x_h, stride_x_w = x.stride()
    stride_out_b, stride_out_c, _, stride_out_h, stride_out_w = out.stride()

    # Grid dimensions: (B, H, W)
    grid = (B, H, W)

    # BLOCK_SIZE_C is a tunable parameter, typically a power of 2
    # We choose the next power of 2 greater than or equal to C
    BLOCK_SIZE_C = triton.next_power_of_2(C) if C > 0 else 1

    softmax_tanh_scale_kernel[grid](
        x, out,
        C,
        scaling_factor,
        stride_x_b, stride_x_c, stride_x_h, stride_x_w,
        stride_out_b, stride_out_c, stride_out_h, stride_out_w,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a series of operations:
    1. Transposed 3D convolution (PyTorch)
    2. Fused Mean pooling (across depth) + Addition (Triton kernel)
    3. Fused Softmax (across channels) + Tanh activation + Scaling (Triton kernel)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Bias is a learnable parameter, initialized randomly
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1, dtype=torch.float32))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        # Ensure input is FP32 for Triton kernels
        x = x.to(torch.float32)

        # 1. Transposed 3D convolution (PyTorch operator)
        x = self.conv_transpose(x)                            # (B, C_out, D, H, W)

        # 2. Fused Mean pool over depth dim (D) and Bias add per channel (Triton kernel)
        x = triton_mean_pool_add_bias(x, self.bias)           # (B, C_out, 1, H, W)

        # 3. Fused Softmax over channels, Tanh nonlinearity, and Scaling (Triton kernel)
        x = triton_softmax_tanh_scale(x, self.scaling_factor) # (B, C_out, 1, H, W)
        return x