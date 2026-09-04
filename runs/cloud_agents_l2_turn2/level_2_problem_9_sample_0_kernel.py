import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_sub_mul_relu_kernel(
    x_ptr,  # Pointer to input tensor
    out_ptr,  # Pointer to output tensor
    n_elements,  # Total number of elements
    subtract_val,  # Scalar value to subtract
    multiply_val,  # Scalar value to multiply
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate the starting offset for the current program
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Generate a range of offsets within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle elements near the end of the tensor
    mask = offsets < n_elements

    # Load input values, ensuring FP32 precision
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Perform fused operations: subtraction, multiplication, and ReLU
    x = x - subtract_val
    x = x * multiply_val
    out = tl.maximum(x, 0.0)  # ReLU activation

    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_sub_mul_relu(x: torch.Tensor, subtract_value: float, multiply_value: float):
    """
    This function wraps the Triton kernel call for fused subtraction, multiplication, and ReLU.
    It:
      1. Ensures the input is contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    """
    assert x.is_cuda, "Input tensor must be on CUDA."
    x = x.contiguous()  # Ensure contiguous memory layout for efficient access

    # Prepare an empty output tensor with the same shape and dtype as input
    out = torch.empty_like(x)

    # Total number of elements in the tensor
    n_elements = x.numel()

    # Triton block size - a tunable parameter for optimal performance.
    # For element-wise operations, 1024 or 2048 are common choices.
    BLOCK_SIZE = 1024

    # Determine the grid (number of blocks) needed to cover all elements
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    fused_sub_mul_relu_kernel[grid](
        x,
        out,
        n_elements,
        subtract_value,
        multiply_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    """
    Optimized Model that performs a matrix multiplication using PyTorch's nn.Linear,
    followed by a fused subtraction, multiplication, and ReLU activation using a custom Triton kernel.
    This fusion reduces memory traffic and kernel launch overheads for the element-wise operations.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # Store these scalar values as attributes; they will be passed to the Triton kernel
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Step 1: Perform the matrix multiplication and bias addition using PyTorch's highly optimized Linear layer.
        x = self.linear(x)
        
        # Step 2: Apply the fused element-wise operations (subtraction, multiplication, ReLU)
        # using our custom Triton kernel for efficiency.
        x = triton_fused_sub_mul_relu(x, self.subtract_value, self.multiply_value)
        return x