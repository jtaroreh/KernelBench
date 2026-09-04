import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def sum_reduction_dim_kernel(
    x_ptr,
    out_ptr,
    outer_size,
    reduce_size,
    inner_size,
    BLOCK_REDUCE: tl.constexpr,
):
    pid_outer = tl.program_id(0)
    pid_inner = tl.program_id(1)
    
    acc = tl.zeros([BLOCK_REDUCE], dtype=tl.float32)
    offsets_reduce = tl.arange(0, BLOCK_REDUCE)
    
    for r_start in range(0, reduce_size, BLOCK_REDUCE):
        r_offsets = r_start + offsets_reduce
        mask = r_offsets < reduce_size
        idx = pid_outer * (reduce_size * inner_size) + r_offsets * inner_size + pid_inner
        val = tl.load(x_ptr + idx, mask=mask, other=0.0)
        acc += val
    
    total = tl.sum(acc, axis=0)
    out_idx = pid_outer * inner_size + pid_inner
    tl.store(out_ptr + out_idx, total)

def triton_reduce_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    assert x.is_cuda, "Tensor must be on CUDA."
    x = x.contiguous()
    dim = dim if dim >= 0 else dim + x.ndim
    shape = list(x.shape)
    outer_size = 1
    for s in shape[:dim]:
        outer_size *= s
    reduce_size = shape[dim]
    inner_size = 1
    for s in shape[dim+1:]:
        inner_size *= s
        
    out_shape = shape[:dim] + shape[dim+1:]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    BLOCK_REDUCE = min(1024, triton.next_power_of_2(reduce_size))
    grid = (outer_size, inner_size)
    sum_reduction_dim_kernel[grid](x, out, outer_size, reduce_size, inner_size, BLOCK_REDUCE=BLOCK_REDUCE)
    return out
