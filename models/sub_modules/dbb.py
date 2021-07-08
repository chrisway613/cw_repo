"""Implements 6 transformations mentioned in DBB(Diverse Branch Block).
   For a close look, pls see: https://arxiv.org/pdf/2103.13425.pdf"""

import torch
import torch.nn as nn


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """conv接bn模块等效融合成1个conv"""

    conv_weight = conv.weight
    # 由于conv后接bn，因此conv中没有设置bias
    bias = 0

    gamma = bn.weight
    beta = bn.bias

    mean = bn.running_mean
    var = bn.running_var
    # 设立极小值下限，防止方差过小时进行除法操作而导致溢出
    eps = bn.eps
    # 基于方差和极小值计算标准差
    std = (var + eps).sqrt()

    # 基于conv和bn的参数值计算出等效融合后的卷积参数值
    fused_conv_weight = (gamma / std).reshape(-1, 1, 1, 1) * conv_weight
    fused_conv_bias = gamma * (bias - mean) / std + beta
    # 将计算好的参数值设置到参数字典中
    param_dict = {'weight': fused_conv_weight, 'bias': fused_conv_bias}

    # 等效融合而成的卷积
    fused_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
    )
    # 设置参数值
    for name, param in fused_conv.named_parameters():
        # 使用字典的get()方法，在参数名称对不上时可以保留原参数值
        assert param_dict.get(name) is not None
        param.data = param_dict[name]
    # 也可使用以下语句
    # fused_conv.weight.data = fused_conv_weight
    # fused_conv.bias.data = fused_conv_bias

    return fused_conv


def test_fuse_conv_bn():
    # 由于conv后面接bn，因此设置bias=False
    conv = nn.Conv2d(3, 3, 3, bias=False)
    bn = nn.BatchNorm2d(3)

    '''注意要设置验证模式(eval)，为的是固定住bn每次前向过程中计算出的均值与方差，以方便验证'''

    # 原始的conv接bn模块
    mod = nn.Sequential(conv, bn)
    mod.eval()

    # 将conv接bn模块等效融合成1个conv
    fused_conv = fuse_conv_bn(conv, bn)
    fused_conv.eval()

    x = torch.randn(1, 3, 7, 7)
    y1 = mod(x)
    y2 = fused_conv(x)

    # 验证等效性
    is_match = torch.allclose(y1, y2)
    print(f"y1==y2: {is_match}")


def fuse_multi_conv_add(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """多分支conv输出相加模块等效融合成1个conv"""

    # 首先验证各分支中conv配置的一致性
    assert conv1.in_channels == conv2.in_channels
    assert conv1.out_channels == conv2.out_channels
    assert conv1.kernel_size == conv2.kernel_size
    assert conv1.stride == conv2.stride
    assert conv1.padding == conv2.padding
    assert conv1.dilation == conv2.dilation
    assert conv1.groups == conv2.groups

    # 计算出融合后conv的参数值，并存储在参数字典中
    fused_conv_weight = conv1.weight + conv2.weight
    fused_conv_bias = conv1.bias + conv2.bias
    param_dict = {'weight': fused_conv_weight, 'bias': fused_conv_bias}

    # 将参数值设置到融合后的conv中
    fused_conv = nn.Conv2d(
        conv1.in_channels, conv1.out_channels, conv1.kernel_size,
        stride=conv1.stride, padding=conv1.padding, dilation=conv1.dilation, groups=conv1.groups
    )
    for name, param in fused_conv.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_conv


def test_fuse_multi_conv_add():
    class BranchAdd(nn.Module):
        """实现一个module，表示多分支conv相加"""

        def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d):
            super().__init__()

            self.conv1 = conv1
            self.conv2 = conv2

        def forward(self, x):
            conv1_out = self.conv1(x)
            conv2_out = self.conv2(x)
            assert conv1_out.shape == conv2_out.shape

            return conv1_out + conv2_out

    conv1 = nn.Conv2d(3, 2, 3)
    conv2 = nn.Conv2d(3, 2, 3)
    branch_add_mod = BranchAdd(conv1, conv2)
    fused_conv = fuse_multi_conv_add(conv1, conv2)

    x = torch.randn(1, 3, 5, 5)
    y1 = branch_add_mod(x)
    y2 = fused_conv(x)
    print(f"y1==y2: {torch.allclose(y1, y2)}")


def fuse_seq_conv(conv_1: nn.Conv2d, conv_k: nn.Conv2d) -> nn.Conv2d:
    """将1x1 conv接KxK conv模块等效融合成1个conv"""

    '''1x1 conv的shape是(D,C,1,1); KxK conv的shape是(E,D,K,K)'''

    # 前两个维度转置后的1x1 conv
    # (C,D,1,1)
    transposed_conv_1 = nn.Conv2d(
        conv_1.out_channels, conv_1.in_channels, 1,
        stride=conv_1.stride, padding=conv_1.padding, dilation=conv_1.dilation, groups=conv_1.groups,
        bias=False
    )
    transposed_conv_1.weight.data = conv_1.weight.data.clone().transpose(0, 1)

    # 将KxK conv的卷积核作为特征图
    # (E,D,K,K)
    kernel_k = conv_k.weight.data
    # 使用维度转置后的1x1 conv对KxK conv的卷积核进行卷积
    # 卷积的输出即为融合成的conv的weight
    # (E,C,K,K)
    fused_kernel = transposed_conv_1(kernel_k)

    fused_conv = nn.Conv2d(
        fused_kernel.size(1), fused_kernel.size(0), fused_kernel.shape[2:],
        stride=conv_k.stride, padding=conv_k.padding, dilation=conv_k.dilation, groups=conv_k.groups
    )
    fused_conv.weight.data = fused_kernel
    # conv_k.weight: (E,D,K,K);
    # conv_1.bias: (D,)->(1,D,1,1);
    # conv_k.bias: (E,)
    fused_conv.bias.data = (conv_k.weight.data * conv_1.bias.data.reshape(1, -1, 1, 1)).sum((1, 2, 3)) \
        + conv_k.bias.data

    return fused_conv


def test_fuse_seq_conv():
    conv_1 = nn.Conv2d(3, 2, 1)
    conv_k = nn.Conv2d(2, 4, 3)
    # 1x1 conv接KxK conv序列模块
    seq_mod = nn.Sequential(conv_1, conv_k)
    # 等效融合成的1个conv
    fused_conv = fuse_seq_conv(conv_1, conv_k)

    x = torch.randn(1, 3, 5, 5)
    y1 = seq_mod(x)
    y2 = fused_conv(x)
    print(f"y1==y2:{torch.allclose(y1, y2)}")


def fuse_multi_conv_concat(conv1: nn.Conv2d, conv2: nn.Conv2d) -> nn.Conv2d:
    """多分支conv输出拼接模块等效融合成1个conv"""

    assert conv1.in_channels == conv2.in_channels
    assert conv1.kernel_size == conv2.kernel_size

    conv1_kernel = conv1.weight.data
    conv1_bias = conv1.bias.data

    conv2_kernel = conv2.weight.data
    conv2_bias = conv2.bias.data

    # 将2个conv的参数对应concat即可得到融合的1个conv的参数
    fused_conv_kernel = torch.cat([conv1_kernel, conv2_kernel], dim=0)
    fused_conv_bias = torch.cat([conv1_bias, conv2_bias], dim=0)
    param_dict = {'weight': fused_conv_kernel, 'bias': fused_conv_bias}

    # 等效融合的1个conv
    fused_conv = nn.Conv2d(
        conv1.in_channels, fused_conv_kernel.size(0), conv1.kernel_size,
        stride=conv1.stride, padding=conv1.padding, dilation=conv1.dilation, groups=conv1.groups,
    )
    # 设置参数值
    for name, param in fused_conv.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return fused_conv


def test_fuse_multi_conv_concat():
    class BranchConcat(nn.Module):
        """实现一个module来代表多分支输出拼接模块"""

        def __init__(self, conv1: nn.Conv2d, conv2: nn.Conv2d):
            super().__init__()

            self.conv1 = conv1
            self.conv2 = conv2

        def forward(self, x):
            conv1_out = self.conv1(x)
            conv2_out = self.conv2(x)
            assert conv1_out.shape[1:] == conv2_out.shape[1:]

            return torch.cat([conv1_out, conv2_out], dim=1)

    conv1 = nn.Conv2d(3, 4, 3)
    conv2 = nn.Conv2d(3, 4, 3)
    branch_concat_mod = BranchConcat(conv1, conv2)
    fused_conv = fuse_multi_conv_concat(conv1, conv2)

    x = torch.randn(1, 3, 5, 5)
    y1 = branch_concat_mod(x)
    y2 = fused_conv(x)
    print(f"y1==y2: {torch.allclose(y1, y2)}")


def avg_pool_transform(channels: int, groups: int, avg_pool: nn.AvgPool2d) -> nn.Conv2d:
    """将平均池化等效转换成conv"""

    # 适配分组卷积，每组的输入通道数
    input_dim = channels // groups

    k1, k2 = avg_pool.kernel_size if isinstance(avg_pool.kernel_size, tuple) else [avg_pool.kernel_size] * 2
    # 等效转换的卷积
    resultant_conv = nn.Conv2d(
        channels, channels, (k1, k2),
        stride=avg_pool.stride, padding=avg_pool.padding, groups=groups, bias=False
    )
    assert resultant_conv.weight.shape == torch.Size([channels, input_dim, k1, k2])

    # 先将卷积权重清零
    resultant_conv.weight.data.zero_()
    # 第i个输出通道仅来自于第i个输入通道，模拟avg_pooling的工作方式
    # (深度可分离卷积亦是如此)
    dim0 = torch.arange(channels)
    # (input_dim,)->(input_dim,1)->(input_dim,groups)->(groups,input_dim)->(groups*input_dim)
    dim1 = torch.arange(input_dim).unsqueeze(1).expand(input_dim, groups).T.flatten()
    assert dim0.size() == dim1.size()
    resultant_conv.weight.data[dim0, dim1, :, :] = 1. / (k1 * k2)

    return resultant_conv


def test_avg_pool_transform(groups=1):
    x = torch.randn(1, 4, 5, 5)
    avg_pool = nn.AvgPool2d(3, stride=1)
    conv = avg_pool_transform(4, groups, avg_pool)

    y1 = avg_pool(x)
    y2 = conv(x)
    print(f"y1==y2: {torch.allclose(y1, y2)}")


def transform_multi_scale_conv(conv: nn.Conv2d, target_kernel_size: tuple) -> nn.Conv2d:
    """将多尺度conv等效转换成指定核大小的conv，如：1xK->KxK, Kx1->KxK, 1x1->KxK (K>=1)等"""

    kernel = conv.weight.data
    h, w = kernel.shape[-2:]
    pad_h, pad_w = conv.padding
    # 原来的conv必须是不改变特征图分辨率的，
    # 这样转换成大尺度conv后才有机会使等效性成立(大尺度conv也可以通过padding不改变特征图分辨率)。
    # 否则，会导致转换前后得到的输出特征图的尺寸不一致
    assert pad_h == (h - 1) >> 1 and pad_w == (w - 1) >> 1

    target_h, target_w = target_kernel_size
    assert target_h >= h and target_w >= w

    # 计算目标kernel大小比原来的kernel上下左右各多出多少
    delta_h = (target_h - h) >> 1
    delta_w = (target_w - w) >> 1
    # 计算目标kernel的shape，出入通道数和原来的保持一致
    target_kernel_shape = list(kernel.shape[:2]) + list(target_kernel_size)

    target_kernel = torch.zeros(target_kernel_shape, device=conv.weight.device)
    # 将原来的kernel“塞”到目标kernel的指定位置
    target_kernel[:, :, delta_h:target_h - delta_h, delta_w:target_w - delta_w] = kernel
    param_dict = {'weight': target_kernel, 'bias': conv.bias.data}

    # 计算特征图需要padding多少，从而维持输入输出特征图的尺寸不变
    target_pad_h = (target_h - 1) >> 1
    target_pad_w = (target_w - 1) >> 1
    target_conv = nn.Conv2d(
        conv.in_channels, conv.out_channels, target_kernel_size,
        stride=conv.stride, padding=(target_pad_h, target_pad_w), dilation=conv.dilation, groups=conv.groups
    )
    for name, param in target_conv.named_parameters():
        assert param_dict.get(name) is not None
        param.data = param_dict[name]

    return target_conv


def test_transform_multi_scale_conv(kernel_size: tuple = (1, 1), target_kernel_size: tuple = (3, 3)):
    h, w = kernel_size
    pad_h = (h - 1) >> 1
    pad_w = (w - 1) >> 1

    # 原先的conv
    conv = nn.Conv2d(3, 4, kernel_size, padding=(pad_h, pad_w))
    # 转换后的conv
    transformed_conv = transform_multi_scale_conv(conv, target_kernel_size)

    x = torch.randn(1, 3, 5, 5)
    y1 = conv(x)
    y2 = transformed_conv(x)
    print(f"y1==y2: {torch.allclose(y1, y2)}")


if __name__ == '__main__':
    # test_fuse_conv_bn()
    # test_fuse_multi_conv_add()
    # test_fuse_seq_conv()
    # test_fuse_multi_conv_concat()
    # test_avg_pool_transform(groups=2)
    test_transform_multi_scale_conv((3, 1), (3, 3))
