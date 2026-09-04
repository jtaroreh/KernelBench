import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def group_norm_min_kernel(
    X, GN_W, GN_B, MinVals,
    N, C, G, CPG,
    BLOCK_G: tl.constexpr, BLOCK_CPG: tl.constexpr
):
    # Each program handles one row (batch element)
    row_idx = tl.program_id(0)
    
    # Create 2D offsets for (G, CPG)
    g_offsets = tl.arange(0, BLOCK_G)
    cpg_offsets = tl.arange(0, BLOCK_CPG)
    
    # Mask for loading
    g_mask = g_offsets < G
    cpg_mask = cpg_offsets < CPG
    mask = g_mask[:, None] & cpg_mask[None, :]
    
    # Load row data and reshape to (G, CPG)
    # X is (N, C), GN_W and GN_B are (C,)
    offsets = g_offsets[:, None] * CPG + cpg_offsets[None, :]
    x = tl.load(X + row_idx * C + offsets, mask=mask, other=0.0)
    
    # Compute mean and variance per group
    # mean: (G,)
    mean = tl