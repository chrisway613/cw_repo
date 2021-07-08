"""Coordinate Attention with Pytorch Implementation, details see:
   https://github.com/Andrew-Qibin/CoordAttention"""

import torch
import torch.nn as nn


class CoordAttn(nn.Module):
    def __init__(self, h, w, c, reduction=16):
        """
        对特征在空间和通道维度上实施注意力。其中，空间维度分治为水平和竖直两个方向分别编码注意力。
        :param h: 输入特征图的高；
        :param w: 输入特征图的宽；
        :param c: 输入特征图的通道数；
        :param reduction: 通道压缩率
        """
        super().__init__()

        # 在x方向平均池化
        self.x_pool = nn.AdaptiveAvgPool2d((h, 1))
        # 在y方向上平均池化
        self.y_pool = nn.AdaptiveAvgPool2d((1, w))

        # 通道压缩
        self.reduce = nn.Conv2d(c, c // reduction, 1, bias=False)
        self.bn = nn.BatchNorm2d(c // reduction)
        self.relu = nn.ReLU()

        # 通道扩展(复原)
        self.span_h = nn.Conv2d(c // reduction, c, 1)
        self.span_w = nn.Conv2d(c // reduction, c, 1)

    def forward(self, x):
        h, w = x.shape[2:]

        '''生成两个方向的空间注意力'''
        # (b,c,h,w)->(b,c,h,1)->(b,c,1,h)
        x_h_pool = self.x_pool(x).permute(0, 1, 3, 2)
        # (b,c,h,w)->(b,c,1,w)
        x_w_pool = self.y_pool(x)

        '''通道压缩'''
        # (b,c,1,h+w)
        x_hw_concat = torch.cat([x_h_pool, x_w_pool], dim=-1)
        # (b,c//r,1,h+w)
        x_reduce = self.relu(self.bn(self.reduce(x_hw_concat)))
        # (b,c//r,1,h), (b,c//r,1,w)
        x_h, x_w = x_reduce.split([h, w], dim=-1)

        '''通道扩展(复原)'''
        # (b,c//r,1,h)->(b,c,1,h)->(b,c,h,1)
        x_h_span = self.span_h(x_h).permute(0, 1, 3, 2)
        # (b,c//r,1,w)->(b,c,1,w)
        x_w_span = self.span_w(x_w)

        '''以上通道压缩&扩展是为生成通道注意力，类似SE-Net'''

        # 在y方向上的空间注意力&通道注意力
        weight_h = torch.sigmoid(x_h_span)
        # 在x方向上的空间注意力&通道注意力
        weight_w = torch.sigmoid(x_w_span)
        out = x * weight_h * weight_w

        return out


if __name__ == '__main__':
    ca = CoordAttn(16, 32, 128)
    inputs = torch.randn(1, 128, 16, 32)
    outputs = ca(inputs)
    print(inputs.shape == outputs.shape)
