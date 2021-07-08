"""Pytorch implementation of Involution operator, details see:
   Inverting the Inference of Convolution for Visual Recognition
   (https://arxiv.org/abs/2103.06255)"""

import torch
import torch.nn as nn


class Involution(nn.Module):
    def __init__(self, channels, kernel_size=7, stride=1, group_channels=16, reduction_ratio=4):
        super().__init__()
        assert not (channels % group_channels or channels % reduction_ratio)

        # in_c=out_c
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 每组多少个通道
        self.group_channels = group_channels
        self.groups = channels // group_channels

        # reduce channels
        self.reduce = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, 1),
            nn.BatchNorm2d(channels // reduction_ratio),
            nn.ReLU()
        )
        # span channels
        self.span = nn.Conv2d(
            channels // reduction_ratio,
            self.groups * kernel_size ** 2,
            1
        )

        self.down_sample = nn.AvgPool2d(stride) if stride != 1 else nn.Identity()
        self.unfold = nn.Unfold(kernel_size, padding=(kernel_size - 1) // 2, stride=stride)

    def forward(self, x):
        # Note that 'h', 'w' are height & width of the output feature.

        # generate involution kernel: (b,G*K*K,h,w)
        weight_matrix = self.span(self.reduce(self.down_sample(x)))
        b, _, h, w = weight_matrix.shape

        # unfold input: (b,C*K*K,h,w)
        x_unfolded = self.unfold(x)
        # (b,C*K*K,h,w)->(b,G,C//G,K*K,h,w)
        x_unfolded = x_unfolded.view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)

        # (b,G*K*K,h,w) -> (b,G,1,K*K,h,w)
        weight_matrix = weight_matrix.view(b, self.groups, 1, self.kernel_size ** 2, h, w)
        # (b,G,C//G,h,w)
        mul_add = (weight_matrix * x_unfolded).sum(dim=3)
        # (b,C,h,w)
        out = mul_add.view(b, self.channels, h, w)

        return out


if __name__ == '__main__':
    inputs = torch.randn(1, 64, 14, 14)
    c = inputs.size(1)
    involution = Involution(c, stride=2)
    outputs = involution(inputs)
    # it should be (1,64,7,7)
    print(outputs.shape)
