import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def softmax_sigmoid_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # Parallelize over the spatial dimensions (D*H*W)
    # Each program handles one spatial location across all channels
    pid = tl.program_id(0)
    
    # Offsets for the channels
    channel_offsets = tl.arange(0, BLOCK_SIZE)
    mask = channel_offsets < n_channels
    
    # Load input for this spatial location
    # x is (N, C, D, H, W), we treat it as (N*D*H*W, C)
    offsets = pid * n_channels + channel_offsets
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    
    # Softmax
    z = x - tl.max(x, axis=0)
    numerator = tl.exp(z)
    denominator = tl.sum(numerator, axis=0)
    softmax_out = numerator / denominator
    
    # Sigmoid
    # sigmoid(x) = 1 / (1 + exp(-x))
    sigmoid_out = 1.0 / (1.0 + tl.exp(-softmax_out))
    
    tl.store(out_ptr + offsets, sigmoid_out, mask=mask)

def triton_softmax_sigmoid(x: torch.Tensor):
    # x shape: (N, C, D, H, W)
    N, C, D, H, W = x.shape
    x_flat = x.permute(0, 2, 3, 4, 1).reshape(-1, C)
    out_flat = torch.empty_like(x_flat)
    
    n_spatial = x_flat.shape[0]
    BLOCK_SIZE = triton.next_power_of_2(C)
    
    grid = (n_spatial,)
    softmax_sigmoid_kernel[grid](x_flat, out_flat, x_flat.numel(), C, BLOCK_SIZE=BLOCK_SIZE)
    
    return out_flat.view(N, D, H, W, C).permute(0, 4, 1, 2, 3)

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=bias)

    def forward(self, x):
        x = self.conv_transpose(x)
        # Fusing Softmax and Sigmoid into a single Triton kernel
        return triton_softmax_sigmoid(x)