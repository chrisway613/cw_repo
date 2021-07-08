import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common_tools.random_seed import setup_seed


setup_seed()


def patch_projection(conv_inputs: torch.Tensor, out_channels: int, patch_size: int = 16,
                     absolute: float = 1e-8, relative: float = 1e-5):
    # absolute和relative是torch.allclose()的参数，代表认为两个变量相等的绝对误差和相对误差值，这里可以适当设大一些

    bs, in_channels, h, w = conv_inputs.size()

    # 卷积核权重与偏置
    conv_weights = torch.randn(out_channels, in_channels, patch_size, patch_size)
    bias = torch.randn(out_channels)
    # (bs,out_c,h//ps,w//ps)
    conv_out = F.conv2d(conv_inputs, conv_weights, bias, stride=patch_size)
    print(f"Convolution Outputs shape: {conv_out.shape}")

    # (bs,in_c,h,w)->(bs,in_c,h//ps,ps,w//ps,ps)
    # ->(bs,h//ps,w//ps,in_c,ps,ps)->(bs,(h*W)//(ps**2),in_c*(ps**2))
    # reshape后dim1对应1个通道内的patch数量，这样后续和MLP权重操作时就能够实现同一通道内各个patch参数共享
    mlp_inputs = conv_inputs.view(bs, in_channels, h // patch_size, patch_size,
                                  w // patch_size, patch_size).permute(0, 2, 4, 1, 3, 5).reshape(
        bs, -1, in_channels * patch_size ** 2)
    # MLP权重
    # 对于输入的每个通道，只有patch size ** 2 个不同的权重，
    # 也就是同一通道内各patch的权重一致，从而仿照了卷积的参数共享
    # 并且，各个patch仅和patch内的元素相加权计算，仿照了卷积的局部性
    mlp_weights = conv_weights.view(out_channels, -1).transpose(0, 1)
    # 矩阵乘法
    # (bs,(h*W)//(ps**2),out_c)
    mlp_out = mlp_inputs @ mlp_weights + bias
    print(f"MLP Outputs shape: {mlp_out.shape}")

    mlp_out_reshaped = mlp_out.transpose(1, 2)
    conv_out_reshaped = conv_out.flatten(start_dim=2)

    print(f"conv_out == mlp_out (with reshape): "
          f"{torch.allclose(conv_out_reshaped, mlp_out_reshaped, atol=absolute, rtol=relative)}")
    print(f"equivalence threshold={absolute + relative * mlp_out_reshaped.abs().sum()}")
    print(f"current error={(conv_out_reshaped - mlp_out_reshaped).abs().sum()}")


def token_mixing(mlp_inputs: torch.Tensor, out_hidden_dim: int,
                 absolute: float = 1e-8, relative: float = 1e-5):
    bs, c, n_patches = mlp_inputs.size()

    # MLP的权重和偏置
    mlp_weights = torch.randn(n_patches, out_hidden_dim)
    mlp_bias = torch.randn(out_hidden_dim)

    # (bs,c,out_hidden_dim)
    mlp_out = mlp_inputs @ mlp_weights + mlp_bias
    print(f"MLP Outputs shape: {mlp_out.shape}")
    # print(f"mlp_out:\n{mlp_out}\n")

    h = w = int(math.sqrt(n_patches))
    # 将token结构化成2维
    conv_inputs = mlp_inputs.view(bs, c, h, w)

    # 卷积权重 这里仿照深度可分离卷积 因为各个入通道的计算结果不相加
    # 不同的是，各个如通道参数共享，仿照MLP的token mixing做法：
    # 各通道的权重一样，但其中各空间位置(patches)的权重不一样
    # 于是这里需要repeat操作
    # (n_pathes,out_hidden_dim)->(out_hidden_dim*c,n_patches)->(out_hidden_dim*c,1,h,w)
    conv_weights = mlp_weights.transpose(0, 1).repeat(c, 1).view(-1, 1, h, w)
    # (out_hidden_dim*c)
    conv_bias = mlp_bias.repeat(c)

    # 分组卷积，组数等于入通道数
    # (bs,out_hidden_dim*c,1,1)
    conv_out = F.conv2d(conv_inputs, conv_weights, conv_bias, groups=c)
    print(f"Convolution Outputs shape: {conv_out.shape}")
    # print(f"conv_out:\n{conv_out}\n")

    # (bs,out_hidden_dim*c)
    conv_out_reshaped = conv_out.squeeze()
    mlp_out_reshaped = mlp_out.flatten(start_dim=1)
    print(f"conv_out == mlp_out (with reshape): "
          f"{torch.allclose(conv_out_reshaped, mlp_out_reshaped, atol=absolute, rtol=relative)}")
    print(f"equivalence threshold={absolute + relative * mlp_out.abs().sum()}")
    print(f"current error={(conv_out_reshaped - mlp_out_reshaped).abs().sum()}")


def channel_mixing(mlp_inputs: torch.Tensor, out_channels: int, absolute: float = 1e-8, relative: float = 1e-5):
    bs, n_patches, c = mlp_inputs.size()

    # MLP权重和偏置
    mlp_weights = torch.randn(c, out_channels)
    bias = torch.randn(out_channels)
    # (bs,n_patches,out_channels)
    mlp_out = mlp_inputs @ mlp_weights + bias
    print(f"MLP Outputs shape: {mlp_out.shape}")

    h = w = int(math.sqrt(n_patches))
    conv_inputs = mlp_inputs.transpose(1, 2).view(bs, c, h, w)
    conv_weights = mlp_weights.transpose(0, 1).view(out_channels, c, 1, 1)
    # (bs,out_channels,h,w)
    conv_out = F.conv2d(conv_inputs, conv_weights, bias)
    print(f"Convolution Output shape: {conv_out.shape}")

    mlp_out_reshaped = mlp_out.transpose(1, 2)
    conv_out_reshaped = conv_out.view(bs, out_channels, -1)
    print(f"conv_out == mlp_out (with reshape): "
          f"{torch.allclose(conv_out_reshaped, mlp_out_reshaped, atol=absolute, rtol=relative)}")
    print(f"equivalence threshold={absolute + relative * mlp_out.abs().sum()}")
    print(f"current error={(conv_out_reshaped - mlp_out_reshaped).abs().sum()}")


class MlpBlock(nn.Module):
    """Token-Mixing和Channel-Mixing的构成：两层全连接，中间使用GELU作为激活函数"""

    def __init__(self, dim, mlp_dim, act_layer=nn.GELU):
        super(MlpBlock, self).__init__()

        self.fc1 = nn.Linear(dim, mlp_dim)
        self.activation = act_layer()
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class MixerBlock(nn.Module):
    def __init__(self, num_patches, hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048, act_layer=nn.GELU):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.token_mixing = MlpBlock(num_patches, tokens_mlp_dim, act_layer=act_layer)

        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.channel_mixing = MlpBlock(hidden_dim, channels_mlp_dim, act_layer=act_layer)

    def forward(self, x):
        y = self.norm1(x)
        # (bs,num_patches,hidden_dim)->(bs,hidden_dim,num_patches)
        y = y.transpose(1, 2)
        y = self.token_mixing(y)
        # (bs,hidden_dim,num_patches)->(bs,num_patches,hidden_dim)
        y = y.transpose(1, 2)
        # (bs,num_patches,hidden_dim)
        x += y

        y = self.norm2(x)
        # (bs,num_patches,hidden_dim)
        y = self.channel_mixing(y)
        x += y

        # (bs,num_patches,hidden_dim)
        return x


class MlpMixer(nn.Module):
    def __init__(self, in_channels=3, input_size=224, num_classes=1000, num_blocks=8, patch_size=16,
                 hidden_dim=512, tokens_mlp_dim=256, channels_mlp_dim=2048):
        super().__init__()

        num_patches = int((input_size // patch_size) ** 2)

        self.patch_projection = nn.Conv2d(in_channels, hidden_dim, patch_size, stride=patch_size)
        self.blocks = nn.Sequential(
            *[MixerBlock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # Patch Projection
        # (bs,hidden_dim,h // patch_size,w // patch_size)
        x = self.patch_projection(x)

        bs, hidden_dim = x.shape[:2]
        # (bs,num_patches,hidden_dim)
        x = x.view(bs, hidden_dim, -1).transpose(1, 2)

        # N x Mixer-Layers
        # (bs,num_patches,hidden_dim)
        x = self.blocks(x)

        # Layer Norm
        x = self.norm(x)
        # Global Average Pooling
        # (bs,num_patches,hidden_dim)->(bs,hidden_dim)
        x = x.mean(1)

        # Classification Head
        # (bs,hidden_dim)->(bs,num_classes)
        return self.classifier(x)

    def _init_weights(self):
        for n, mod in self.named_modules():
            if isinstance(mod, nn.LayerNorm):
                nn.init.ones_(mod.weight)
                nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Conv2d):
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)
            elif isinstance(mod, nn.Linear):
                # 头部初始化为0
                if n.startswith('classifier'):
                    nn.init.zeros_(mod.weight)
                    nn.init.zeros_(mod.bias)
                else:
                    nn.init.xavier_uniform_(mod.weight)

                    if mod.bias is not None:
                        # token mixing 和 channel mixing的偏置使用正态分布初始化
                        if n.endswith('mixing'):
                            nn.init.normal_(mod.bias, std=1e-6)
                        # 其余mlp的偏置均初始化为0
                        else:
                            nn.init.zeros_(mod.bias)


if __name__ == '__main__':
    # batch_size = 2
    # in_c, out_c = 3, 4
    #
    # '------------------patch projection------------------'
    #
    # x = torch.randn(batch_size, in_c, 48, 48)
    # patch_projection(x, out_c, absolute=1e-5)
    #
    # '--------------------token mixing----------------------'
    #
    # channels = 4
    # num_patches = 16
    # out_dim = 6
    #
    # tokens = torch.randn(2, channels, num_patches)
    # token_mixing(tokens, out_dim)
    #
    # '--------------------channel mixing----------------------'
    #
    # channel_mixing(tokens, out_dim)

    '--------------------Mlp-Mixer Forward-------------------'

    bs = 1
    in_c = 3
    input_size = 224
    img = torch.randn(bs, in_c, input_size, input_size)
    mlp_mixer = MlpMixer(in_channels=in_c)
    outputs = mlp_mixer(img)
    print(outputs.shape)
