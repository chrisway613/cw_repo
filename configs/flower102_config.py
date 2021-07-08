# -*- coding: utf-8 -*-

"""
# @file name  : flower102_config.py
# @author     : CW
# @date       : 2021-06-29
# @brief      : 花朵分类参数配置
"""

import torch

from easydict import EasyDict
from torchvision import transforms


# 访问属性的方式去使用key-value 即通过 .key获得value
cfg = EasyDict()

cfg.data_root_dir = '/home/cw/OpenSources/Datasets/102Flowers/data/oxford-102-flowers'

# 是否使用类别均衡采用
cfg.cb = False
# 是否采用渐进式采样
cfg.pb = False
# 是否采用mixup
cfg.mixup = False
# beta分布的参数. beta分布是一组定义在(0,1) 区间的连续概率分布。
cfg.mixup_alpha = 1.

# 是否采用标签平滑
cfg.label_smooth = False
# 标签平滑超参数 eps
cfg.label_smooth_eps = 1e-2

# 是否使用在线困难样本挖掘
cfg.ohem = False

cfg.train_bs = 32
cfg.valid_bs = 32
# 经验法则，num_workers = 4 * GPUs
cfg.num_workers = 4 * torch.cuda.device_count()

cfg.lr = 1e-2
cfg.momentum = .9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [30, 45]
cfg.max_epoch = 50
cfg.log_interval = 10

# backbone预训练权重
cfg.pretrained_weight = ''
# 以往训练好的模型权重
cfg.historical_weight = ''

# 最短边
cfg.S = 256
# 网络输入尺寸
cfg.input_size = 224

# Imagenet 120万图像统计得来的图像均值和标准差
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
# 划分数据集后统计出来的图像均值和标准差(脚本见bins/split_flower_dataset.py)
cfg.mean_dict = {
    "train": [0.433, .287, .376],
    "valid": [.439, .281, .374],
    "test": [.433, .278, .372]
}
cfg.std_dict = {
    "train": [.296, .271, .246],
    "valid": [.301, .264, .244],
    "test": [.300, .268, .244]
}

# 训练集数据增强
cfg.transforms_train = transforms.Compose([
    # 最短边缩放至S，另一边等比例缩放
    transforms.Resize(cfg.S),
    # 中心裁减出SxS大小
    transforms.CenterCrop(cfg.S),
    # 随机裁减
    transforms.RandomCrop(cfg.input_size),
    # 随机水平翻转，默认概率为0.5
    transforms.RandomHorizontalFlip(),
    # 转换成[0,1]张量
    transforms.ToTensor(),
    # 归一化：0均值、1标准差
    # transforms.Normalize(norm_mean, norm_std),
    transforms.Normalize(cfg.mean_dict['train'], cfg.std_dict['train'])
])
# 验证集数据增强
cfg.transforms_valid = transforms.Compose([
    transforms.Resize((cfg.input_size,) * 2),
    transforms.ToTensor(),
    # 归一化：0均值、1标准差
    # transforms.Normalize(norm_mean, norm_std),
    transforms.Normalize(cfg.mean_dict['valid'], cfg.std_dict['valid'])
])

# 是否使用分布式训练
distributed = False
# 使否使用同步BN
use_sync_bn = False
