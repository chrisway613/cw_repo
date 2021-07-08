"""Pytorch implementation of 'RepVGG: Making VGG-style ConvNets Great Again (CVPR-2021)',
   refer to https://github.com/DingXiaoH/RepVGG"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchscan import summary


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    mod = nn.Sequential()
    mod.add_module(
        'conv',
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding, groups=groups, bias=False)
    )
    mod.add_module('bn', nn.BatchNorm2d(out_channels))

    return mod


class SEBlock(nn.Module):
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x = F.avg_pool2d(inputs, kernel_size=inputs.size(3))
        x = self.down(x)
        x = F.relu(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class RepVGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super().__init__()

        # 是否转换成部署状态
        self.deploy = deploy
        # 分组卷积的组数
        self.groups = groups
        self.in_channels = in_channels

        if kernel_size != 3 or padding != 1:
            raise ValueError(f"kernel size != 3 or padding != 1 is not supported")

        self.nonlinearity = nn.ReLU()
        self.se = SEBlock(out_channels, out_channels // 16) if use_se else nn.Identity()

        if deploy:
            # 部署状态就用一个融合后的3x3卷积
            self.rbr_reparam = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)
        else:
            self.rbr_dense = conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels, out_channels, 1, stride, 0, groups=groups)
            # 仅当输入输出的shape完全一致时，才使用identity分支
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            print(f'RepVGGBlock identity = {self.rbr_identity}\n')

    def forward(self, inputs):
        # 部署状态，直接经过一个等效融合后的3x3卷积即可
        if hasattr(self, 'rbr_reparam'):
            return self.rbr_reparam(inputs)

        # 训练状态
        id_out = 0 if self.rbr_identity is None else self.rbr_identity(inputs)
        # 3个分支element-wise add
        fusion = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out

        return self.nonlinearity(self.se(fusion))

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel_1x1):
        """将1x1conv填充为3x3conv，其中1x1conv的权重在3x3conv中间。"""

        if kernel_1x1 is None:
            return 0
        else:
            return F.pad(kernel_1x1, [1] * 4)

    def _fuse_bn_tensor(self, branch):
        """将各分支(3x3conv_bn、1x1conv_bn、identity)等效融合成一个卷积"""

        if branch is None:
            return 0, 0

        # 3x3conv+bn 或 1x1conv+bn分支
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight

            gamma = branch.bn.weight
            beta = branch.bn.bias

            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            eps = branch.bn.eps
        # identity分支
        else:
            # 实质上是bn
            assert isinstance(branch, nn.BatchNorm2d), f"identity branch should be nn.BatchNorm2d, " \
                                                       f"got {type(branch)}"

            if not hasattr(self, 'id_tensor'):
                # 分组卷积场景，输入通道在每组中的数量
                # 也就是每组的卷积核的通道数目
                input_dim = self.in_channels // self.groups
                # 只有第i个kernel的第i个通道权重为1，其余为0，以模仿identity的行为
                kernel_value = torch.zeros((self.in_channels, input_dim, 3, 3))
                for c in range(self.in_channels):
                    # 由于分组卷积场景下，每个kernel仅有input_dim个通道，于是需要取模
                    # 1x1conv的权重放在3x3conv的中心位置
                    kernel_value[c, c % input_dim, 1, 1] = 1

                self.id_tensor = kernel_value.to(branch.weight.device)

            kernel = self.id_tensor

            gamma = branch.weight
            beta = branch.bias

            running_mean = branch.running_mean
            running_var = branch.running_var
            eps = branch.eps

        # 根据公式等效融合计算
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)

        # 第二项看起来和公式不一样，但其实是由于各分支的conv中bias=0，代入公式后即可得如下
        return kernel * t, beta - running_mean * gamma / std

    def get_equivalent_kernel_bias(self):
        # 分别等效融合3x3conv_bn、1x1conv_bn、identity(bn)为一个卷积
        kernel_3x3, bias_3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel_1x1, bias_1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernel_id, bias_id = self._fuse_bn_tensor(self.rbr_identity)

        # 将以上三个融合后的卷积weight和bias对应相加得到最终融合的形式
        # 注意1x1conv的转换结果要先pad为3x3conv
        kernel_fused = kernel_3x3 + self._pad_1x1_to_3x3_tensor(kernel_1x1) + kernel_id
        bias_fused = bias_3x3 + bias_1x1 + bias_id

        return kernel_fused, bias_fused

    def switch_to_deploy(self):
        # 说明已经在部署状态
        if hasattr(self, 'rbr_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(self.rbr_dense.conv.in_channels, self.rbr_dense.conv.out_channels,
                                     self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        # 由于要转换为部署状态，于是取消各参数梯度
        for param in self.parameters():
            param.detach_()

        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        self.__delattr__('rbr_identity')


class RepVGG(nn.Module):
    def __init__(self, num_blocks, num_classes=1000,
                 width_multiplier=None, override_groups_map=None, deploy=False, use_se=False):
        super().__init__()

        # width_multiplier用于各个stage的宽度缩放，共有4个stage，仿照ResNet的layer1~4
        assert len(width_multiplier) == 4, f"length of 'width_multiplier' should be 4"

        # 各层使用的分组卷积的组数，key是网络层索引，value是groups数
        self.override_groups_map = override_groups_map or dict()
        assert 0 not in self.override_groups_map, f"num of groups should not be 0"

        self.deploy = deploy
        self.use_se = use_se

        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.stage0 = RepVGGBlock(3, self.in_planes, 3, stride=2, padding=1, deploy=deploy, use_se=use_se)

        # 当前网络层索引，用于映射各层使用的分组卷积的组数
        self.cur_layer_idx = 1
        # 仿照ResNet的layer1~4，其中每个stage只有第一个block会进行2倍下采样
        self.stage1 = self._make_stage(int(64 * width_multiplier[0]), num_blocks[0], stride=2)
        self.stage2 = self._make_stage(int(128 * width_multiplier[1]), num_blocks[1], stride=2)
        self.stage3 = self._make_stage(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.stage4 = self._make_stage(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(int(512 * width_multiplier[3]), num_classes)

    def forward(self, x):
        out = self.stage0(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        # (B,C,1,1)->(B,C)
        out = self.gap(out).view(out.size(0), -1)
        out = self.linear(out)

        return out

    def _make_stage(self, planes, num_blocks, stride=1):
        # 获取当前网络层对应的分组卷积组数
        cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
        # 只有第一个block可能进行下采样和改变通道数
        blocks = [RepVGGBlock(self.in_planes, planes, 3, stride=stride, padding=1, groups=cur_groups,
                              deploy=self.deploy, use_se=self.use_se)]

        # 注意更新入通道数
        self.in_planes = planes
        # 注意要更新当前网络层的索引
        self.cur_layer_idx += 1

        # 后续的block不改变特征图通道数和分辨率
        for _ in range(1, num_blocks):
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(self.in_planes, planes, 3, padding=1, groups=cur_groups,
                                      deploy=self.deploy, use_se=self.use_se))

            # 注意要更新当前网络层的索引
            self.cur_layer_idx += 1

        return nn.Sequential(*blocks)


optional_groupwise_layers = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]
g2_map = {layer: 2 for layer in optional_groupwise_layers}
g4_map = {layer: 4 for layer in optional_groupwise_layers}


def create_RepVGG_A0(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[0.75, 0.75, 0.75, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A1(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_A2(deploy=False):
    return RepVGG(num_blocks=[2, 4, 14, 1], num_classes=1000,
                  width_multiplier=[1.5, 1.5, 1.5, 2.75], override_groups_map=None, deploy=deploy)


def create_RepVGG_B0(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[1, 1, 1, 2.5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=None, deploy=deploy)


def create_RepVGG_B1g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B1g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2, 2, 2, 4], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B2g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B2g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_B3(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=None, deploy=deploy)


def create_RepVGG_B3g2(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g2_map, deploy=deploy)


def create_RepVGG_B3g4(deploy=False):
    return RepVGG(num_blocks=[4, 6, 16, 1], num_classes=1000,
                  width_multiplier=[3, 3, 3, 5], override_groups_map=g4_map, deploy=deploy)


def create_RepVGG_D2se(deploy=False):
    return RepVGG(num_blocks=[8, 14, 24, 1], num_classes=1000,
                  width_multiplier=[2.5, 2.5, 2.5, 5], override_groups_map=None, deploy=deploy, use_se=True)


func_dict = {
    'RepVGG-A0': create_RepVGG_A0,
    'RepVGG-A1': create_RepVGG_A1,
    'RepVGG-A2': create_RepVGG_A2,
    'RepVGG-B0': create_RepVGG_B0,
    'RepVGG-B1': create_RepVGG_B1,
    'RepVGG-B1g2': create_RepVGG_B1g2,
    'RepVGG-B1g4': create_RepVGG_B1g4,
    'RepVGG-B2': create_RepVGG_B2,
    'RepVGG-B2g2': create_RepVGG_B2g2,
    'RepVGG-B2g4': create_RepVGG_B2g4,
    'RepVGG-B3': create_RepVGG_B3,
    'RepVGG-B3g2': create_RepVGG_B3g2,
    'RepVGG-B3g4': create_RepVGG_B3g4,
    'RepVGG-D2se': create_RepVGG_D2se
}


def get_RepVGG_func_by_name(name):
    return func_dict[name]


#   Use this for converting a RepVGG model or a bigger model with RepVGG as its component
#   Use like this
#   model = create_RepVGG_A0(deploy=False)
#   train model or load weights
#   repvgg_model_convert(model, save_path='repvgg_deploy.pth')
#   If you want to preserve the original model, call with do_copy=True

# ====================== for using RepVGG as the backbones of a bigger model, e.g., PSPNet,
# the pseudo code will be like
# train_backbone = create_RepVGG_B2(deploy=False)
# train_backbone.load_state_dict(torch.load('RepVGG-B2-train.pth'))
# train_pspnet = build_pspnet(backbones=train_backbone)
# segmentation_train(train_pspnet)
# deploy_pspnet = repvgg_model_convert(train_pspnet)
# segmentation_test(deploy_pspnet)
# ===================== example_pspnet.py shows an example

def repvgg_model_convert(model: nn.Module, save_path=None, do_copy=True):
    """将模型中的RepVGGBlock转换成部署模式、保存权重，返回转换后的模型。
       注意，若do_copy=False则原有(转换前)模型不会被保留"""

    # 深拷贝，这样就不会影响原有模型
    if do_copy:
        model = copy.deepcopy(model)

    for module in model.modules():
        # 将RepVGGBlock转换成部署模式(等效融合各branch为一个卷积)
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()

    if save_path is not None:
        torch.save(model.state_dict(), save_path)

    return model


if __name__ == '__main__':
    arch = 'RepVGG-A0'
    model_build_func = get_RepVGG_func_by_name(arch)
    model = model_build_func(deploy=True)

    num_classes = 102
    in_feats = model.linear.in_features
    model.linear = nn.Linear(in_feats, num_classes)

    summary(model, (3, 224, 224))
