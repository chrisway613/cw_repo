"""DarkNet实现"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_layers(cfg: (list, tuple), in_channels=3, bn=True, flag=True):
    """
        从配置参数中构建DarkNet网络层
        :param cfg: list or tuple 配置参数
        :param in_channels: int 输入通道数, 比如RGB彩图则3, 灰度图则为1
        :param bn:  bool 是否使用批正则化
        :param flag: bool 用于变换卷积核参数(kernel_size, padding)
        :return:
    """

    # 由MaxPooling、Conv、BN、LeakyReLU组成
    # 下采样完全由池化层负责
    layers = []
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(2))
        else:
            layers.append(nn.Conv2d(in_channels, v, (1, 3)[flag], padding=int(flag), bias=not bn))
            if bn:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.LeakyReLU(negative_slope=.1, inplace=True))
            # print(in_channels, v)
            in_channels = v

        # 相邻卷积层的核大小不同
        flag = not flag

    return nn.Sequential(*layers)


class DarkNet19(nn.Module):
    """DarkNet19"""

    # M代表最大池化层
    cfg1 = [32, 'M', 64, 'M', 128, 64, 128, 'M', 256, 128, 256]
    cfg2 = ['M', 512, 256, 512, 256, 512]
    cfg3 = ['M', 1024, 512, 1024, 512, 1024]

    def __init__(self, in_channels=3, num_classes=1000, bn=True, include_top=True,
                 pretrained=False, weight_file=None):
        """
            模型结构初始化
            :param num_classes: int 分类类别数目
            :param in_channels: int 输入通道数
            :param bn: bool 是否使用批次正则化
            :param pretrained: bool 是否加载预训练权重
            :param include_top: bool 是否包含头部(1x1卷积、全局平均池化、Softmax)
        """

        super().__init__()

        self.block1 = make_layers(DarkNet19.cfg1, in_channels=in_channels, bn=bn)
        self.block2 = make_layers(DarkNet19.cfg2, in_channels=DarkNet19.cfg1[-1], bn=bn, flag=False)
        self.block3 = make_layers(DarkNet19.cfg3, in_channels=DarkNet19.cfg2[-1], bn=bn, flag=False)

        # 如果包含头部，则使用1x1卷积将输出通道数映射到类别数
        if include_top:
            self.conv = nn.Conv2d(DarkNet19.cfg3[-1], num_classes, 1)
        self.include_top = include_top

        if pretrained:
            assert os.path.exists(weight_file), f"weight file '{weight_file}' is not existed!"
            self._load_weights(weight_file)
        else:
            self._init_weights()

    def forward(self, x):
        feat1 = self.block1(x)
        feat2 = self.block2(feat1)
        feat3 = self.block3(feat2)

        if self.include_top:
            return F.softmax(self.conv(feat3).mean((2, 3)), dim=1)
        else:
            return feat1, feat2, feat3

    def _load_weights(self, weight_file):
        self.load_state_dict({
            k: v for k, v in zip(
                self.state_dict(), torch.load(weight_file, map_location='cpu').values()
            )
        })

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


if __name__ == '__main__':
    net = DarkNet19(num_classes=20)
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.shape)
