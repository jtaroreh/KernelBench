import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_conv_activation_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    C, D, H, W,
    stride_xC, stride_xD, stride_xH, stride_xW,
    stride_outC, stride_outD, stride_outH, stride_outW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute multi-dimensional indices from flat offset
    # Output shape: [N, C, D, H, W]
    HW = H * W
    DHW = D * HW
    CDHW = C * DHW

    n_idx = offsets // CDHW
    rem = offsets % CDHW
    c_idx = rem // DHW
    rem = rem % DHW
    d_idx = rem // HW
    rem = rem % HW
    h_idx = rem // W
    w_idx = rem % W

    # Load x
    x_offsets = n_idx * stride_xC * DHW + c_idx * stride_xC + d_idx * stride_xD + h_idx * stride_xH + w_idx * stride_xW
    # Wait, the stride arguments passed will be:
    # Let's define standard contiguous strides for [N, C, D, H, W]
    # x has shape [N, C, D, H, W]. 
    # Let's simplify by assuming x and out are contiguous.
    # Actually, standard contiguous strides:
    # stride_C = D * H * W
    # stride_D = H * W
    # stride_H = W
    # stride_W = 1
    # But since torch.Tensor can be passed with arbitrary strides, let's just use standard tensor indexing or flatten assuming contiguous or compute using tensor strides.
    # Alternatively, since we can just use linear indexing if x is contiguous, let's make sure inputs are contiguous.

@triton.jit
def elementwise_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    total_elements, C, DHW,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements

    # x shape: [N, C, D, H, W]
    # scale and bias shape: [C, 1, 1, 1], broadcastable along C
    # Channel index can be found by: (idx // DHW) % C
    c_indices = (idx // DHW) % C

    x = tl.load(x_ptr + idx, mask=mask)
    scale = tl.load(scale_ptr + c_indices, mask=mask)
    bias = tl.load(bias_ptr + c_indices, mask=mask)

    x = x * scale
    x = tl.math.tanh(x)
    x = x * bias
    # sigmoid(x) = 1 / (1 + exp(-x))
    # use clamp for numerical stability
    neg_x = tl.clamp(-x, -88.0, 88.0)
    sigmoid_x = 1.0 / (1.0 + tl.exp(neg_x))

    tl.store(out_ptr + idx, sigmoid_x, mask=mask)

class ModelNew(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.full(bias_shape, float(scaling_factor)))
        self.bias = nn.Parameter(torch.randn(bias_shape)) 

    def forward(self, x):
        x = self.conv(x)
        
        # Ensure contiguous for efficient Triton kernel execution
        x = x.contiguous()
        scale = self.scaling_factor.squeeze().contiguous()
        bias = self.bias.squeeze().contiguous()
        
        N, C, D, H, W = x.shape
        DHW = D * H * W
        total_elements = x.numel()
        
        out = torch.empty_like(x)
        
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
        
        elementwise_kernel[grid](
            x, scale, bias, out,
            total_elements, C, DHW,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return out