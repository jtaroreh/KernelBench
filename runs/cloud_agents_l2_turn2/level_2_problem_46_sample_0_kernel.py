import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_post_conv_avgpool_kernel(
    x_ptr, out_ptr,
    sub1, sub2,
    N, C, H_in, W_in,
    H_out, W_out,
    kP: tl.constexpr,
    stride_n, stride_c, stride_h, stride_w,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread computes one output element of the average pool
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = N * C * H_out * W_out
    mask = idx < total_elements
    
    # Map 1D index to 4D output coordinates (N, C, H_out, W_out)
    w_idx = idx % W_out
    h_idx = (idx // W_out) % H_out
    c_idx = (idx // (W_out * H_out)) % C
    n_idx = idx // (W_out * H_out * C)
    
    # Base pointer for the kP x kP window in the input tensor