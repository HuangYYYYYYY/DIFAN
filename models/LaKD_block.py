import torch
import torch.nn as nn
import torch.nn.functional as F

class DLKD_block(nn.Module):
    def __init__(self, in_channels,out_channels, fusion_kernel_size=3):
        super(DLKD_block, self).__init__()
        # Layer Normalization
        self.ln = nn.LayerNorm([out_channels, 1, 1], elementwise_affine=False)

        self.fusion_depthwise = nn.Conv2d(in_channels, in_channels, fusion_kernel_size,
                                          padding=(fusion_kernel_size - 1) // 2, groups=in_channels, bias=False)
        self.fusion_pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fusion_gate = nn.Conv2d(in_channels, in_channels, fusion_kernel_size,
                                     padding=(fusion_kernel_size - 1) // 2, groups=in_channels, bias=False)
        self.W1 = nn.Conv2d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        t0 = self.ln(x)
        t1 = self.fusion_depthwise(t0)+t0
        t2 = self.fusion_pointwise(t1)+t1
        t3 = self.ln(t2)
        gate = F.gelu(self.fusion_gate(t3))
        fusion_out = self.W1(t3) * gate
        fusion_out = fusion_out + x

        return fusion_out