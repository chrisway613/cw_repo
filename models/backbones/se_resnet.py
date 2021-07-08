import torch.nn as nn

from torchvision.models import ResNet
from torch.hub import load_state_dict_from_url


def conv1x1(in_planes, out_planes, stride=1) -> nn.Conv2d:
    """1x1不带偏置(bias)的卷积"""

    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1) -> nn.Conv2d:
    """3x3带填充(padding)而不带偏置(bias)的卷积"""

    return nn.Conv2d(in_planes, out_planes, 3,
                     stride=stride, padding=dilation, groups=groups, dilation=dilation, bias=False)


class SEModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()

        # 将空间维度压缩为1x1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 通道先压缩后恢复并经过Sigmoid映射为权重
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.avgpool(x).view(b, c)
        weight = self.fc(y).view(b, c, 1, 1)
        scaled = weight * x

        return scaled


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(in_planes, planes, stride=stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.se = SEModule(planes * self.expansion, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 通道注意力加权
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_planes, width)
        self.bn1 = norm_layer(width)

        self.conv2 = conv3x3(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = norm_layer(width)

        out_planes = planes * self.expansion
        self.conv3 = conv1x1(width, out_planes)
        self.bn3 = norm_layer(out_planes)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(planes * self.expansion, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        out += identity
        out = self.relu(out)

        return out


def se_resnet18(num_classes=1000):
    return ResNet(SEBasicBlock, (2, 2, 2, 2), num_classes=num_classes)


def se_resnet34(num_classes=1000):
    return ResNet(SEBasicBlock, (3, 4, 6, 3), num_classes=num_classes)


def se_resnet50(num_classes=1000, pretrained=False):
    model = ResNet(SEBottleneck, (3, 4, 6, 3), num_classes=num_classes)
    if pretrained:
        state_dict = load_state_dict_from_url(
            "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl"
        )
        model.load_state_dict(state_dict)

    return model


def se_resnet101(num_classes=1_000):
    return ResNet(SEBottleneck, (3, 4, 23, 3), num_classes=num_classes)


def se_resnet152(num_classes=1_000):
    return ResNet(SEBottleneck, (3, 8, 36, 3), num_classes=num_classes)
