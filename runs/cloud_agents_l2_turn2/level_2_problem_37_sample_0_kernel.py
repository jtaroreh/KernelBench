import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def swish_bias_groupnorm_kernel(
    x_ptr, bias_ptr, gn_weight_ptr, gn_bias_ptr, out_ptr,
    N, C, G, GS, eps,
    BLOCK_G: tl.constexpr, BLOCK_GS: tl.constexpr
):
    """
    Fuses Swish activation, bias addition, and GroupNorm.
    x_ptr: input tensor after matmul (N, C)
    bias_ptr: extra bias parameter (C,)
    gn_weight_ptr: GroupNorm weight (C,)
    gn_bias_ptr: GroupNorm bias (C,)
    out_ptr: output tensor (N, C)
    N: batch_size, C: out_features, G: num_groups, GS: group_size (C // G)
    """
    row_idx = tl.program_id(0)
    
    # Create 2D offsets for groups and elements within groups
    g_offsets = tl.arange(0, BLOCK_G)
    gs_offsets = tl.arange(0, BLOCK_GS)
    
    # Full row offsets: (BLOCK_G, BLOCK_GS)
    # Each row has C elements, divided into G groups of GS elements each.
    offsets = g_offsets