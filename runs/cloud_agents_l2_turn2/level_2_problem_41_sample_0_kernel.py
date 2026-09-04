import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_bn_gelu_relu_kernel(
    x_ptr, weight_ptr, bias_ptr, mean_ptr, var_ptr, out_ptr,
    n_elements, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over all elements in the (batch_size, out_features) tensor
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Calculate column index for each element to access BN parameters
    col_idx = offsets % n_cols

    # Load input and BN parameters
    x = tl.load(x_ptr + offsets, mask=mask)
    weight = tl.load(weight_ptr + col_idx, mask=mask)
    bias = tl.load(bias_ptr + col_idx, mask=mask)
    mean = tl.load(mean_ptr + col_idx, mask=mask)
    var = tl.load(var_ptr + col_idx, mask=mask)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_