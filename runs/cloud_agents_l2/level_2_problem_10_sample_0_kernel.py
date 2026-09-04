import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def hardtanh_mean_tanh_kernel(
    x_ptr,  # Pointer to the input tensor (output of MaxPool2d)
    out_ptr,  # Pointer to the output tensor
    N, C, H_prime, W_prime,  # Dimensions of the input tensor (N, C, H_prime, W_prime)
    min_val, max_val,  # Hardtanh parameters
    BLOCK_SIZE: tl.constexpr,  # Block size for loading elements within a slice
):
    # Each program instance processes one (n, c) slice
    n_c_idx = tl.program_id(0)
    
    # Calculate n and c indices for this program
    n_idx = n_c_idx // C
    c_idx = n_c_idx % C

    # Calculate the total number of elements in one (H_prime, W_prime) slice
    num_elements_per_slice = H_prime * W_prime

    # Calculate the starting pointer for the current (n, c) slice
    # x_ptr is (N, C, H_prime, W_prime)
    # Stride for N: C * H_prime * W_prime
    # Stride for C: H_prime * W_prime
    slice_start_ptr = x_ptr + n_idx * C * num_elements_per_slice + c_idx * num_elements_per_slice

    # Initialize accumulator for sum
    # We use a BLOCK_SIZE array to accumulate sums from different blocks
    # This allows for parallel reduction within the program
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Loop through the elements of the (H_prime, W_prime) slice in blocks
    # Each iteration loads BLOCK_SIZE elements, applies hardtanh, and adds to accumulator
    for block_offset in tl.range(0, num_elements_per_slice, BLOCK_SIZE):
        # Generate offsets for the current block
        offsets = block_offset + tl.arange(0, BLOCK_SIZE)
        
        # Create a mask to handle elements near the end of the slice
        mask = offsets < num_elements_per_slice
        
        # Load elements from the input slice
        x_block = tl.load(slice_start_ptr + offsets, mask=mask, other=0.0)
        
        # Apply Hardtanh activation
        hardtanh_block = tl.maximum(min_val, tl.minimum(max_val, x_block))
        
        # Accumulate the hardtanh-activated values
        # Only add valid elements (where mask is true)
        acc += tl.where(mask, hardtanh_block, 0.0)

    # Perform reduction across the accumulated blocks to get the total sum for the slice
    total_sum = tl.sum(acc, axis=0)

    # Calculate the mean
    mean_val = total_sum / num_elements_per_slice

    # Apply Tanh activation
    tanh_val = tl.tanh(mean_val)

    # Store the result to the output tensor
    # Output tensor is (N, C, 1, 1), so the offset is n_idx * C + c_idx
    out_offset = n_idx * C + c_idx
    tl.store(out_ptr + out_offset, tanh_val)


def triton_hardtanh_mean_tanh(x: torch.Tensor, min_val: float, max_val: float):
    """
    This function wraps the Triton kernel call for fused Hardtanh, Mean, and Tanh.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    assert x.dim() == 4, "Input tensor must be 4D (N, C, H, W)."
    
    # Ensure input is contiguous for efficient memory access
    x = x.contiguous()

    N, C, H_prime, W_prime = x.shape

    # Prepare output tensor with shape (N, C, 1, 1)
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    # Number of programs needed: one for each (n, c) slice
    n_programs = N * C

    # Tunable parameter for block size within each program's reduction loop.
    # This should be a power of 2. For H_prime * W_prime = 128*128 = 16384,
    # BLOCK_SIZE=1024 means 16 iterations, which is efficient.
    BLOCK_SIZE = 1024 

    # Define the grid for the kernel launch. Each program processes one (n, c) slice.
    grid = lambda meta: (n_programs,)

    # Launch the Triton kernel
    hardtanh_mean_tanh_kernel[grid](
        x, out,
        N, C, H_prime, W_prime,
        min_val, max_val,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out


# Original model parameters (for reference)
batch_size = 128
in_channels  = 64  
out_channels = 64  
height = width = 256  
kernel_size  = 3
stride = 1
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1


class ModelNew(nn.Module):
    """
    Optimized Model that replaces Hardtanh, Mean, and Tanh with a custom Triton kernel.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        # Store hardtanh parameters to pass to the Triton kernel
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        # Call the custom Triton-based fused operation
        x = triton_hardtanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
        return x


def get_inputs():
    # randomly generate input tensors based on the model architecture
    return [torch.rand(batch_size, in_channels, height, width).cuda()]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]