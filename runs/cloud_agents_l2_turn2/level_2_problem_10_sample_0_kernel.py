import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def hardtanh_mean_tanh_kernel(
    x_ptr,
    out_ptr,
    N, C, H_prime, W_prime,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program instance processes one (n, c) slice
    n_c_idx = tl.program_id(0)
    
    n_idx = n_c_idx // C
    c_idx = n_c_idx % C

    num_elements_per_slice = H_prime * W_prime
    slice_start_ptr = x_ptr + (n_idx * C + c_idx) * num_elements_per_slice

    # Accumulate sum in float32
    acc = 0.0

    # Loop through the elements of the (H_prime, W_prime) slice
    for block_offset in range(0, num_elements_per_slice, BLOCK_SIZE):
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        mask = offsets < num_elements_per_slice
        
        x_block = tl.load(slice_start_ptr + offsets, mask=mask, other=0.0)
        
        # Apply Hardtanh activation
        hardtanh_block = tl.maximum(min_val, tl.minimum(max_val, x_block))
        
        # Accumulate
        acc += tl.sum(tl.where(mask, hardtanh_block, 0.0))

    # Calculate the mean
    mean_val = acc / num_elements_per_slice

    # Apply Tanh activation using tl.math.tanh
    tanh_val = tl.math.tanh(mean_val)

    # Store the result to the output tensor
    out_offset = n_idx * C + c_idx
    tl.store(out_ptr + out_offset, tanh_val)


def triton_hardtanh_mean_tanh(x: torch.Tensor, min_val: float, max_val: float):
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()
    N, C, H_prime, W_prime = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    n_programs = N * C
    BLOCK_SIZE = 1024 
    grid = (n_programs,)

    hardtanh_mean_tanh_kernel[grid](
        x, out,
        N, C, H_prime, W_prime,
        min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that replaces Hardtanh, Mean, and Tanh with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        # Call the custom Triton-based fused operation
        x = triton_hardtanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
        return x