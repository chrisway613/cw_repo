"""模型训练pipeline：包括数据读取、模型前向预测、反向传播loss、优化器更新参数、学习率更新、记录必要信息、打印日志等"""

import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from easydict import EasyDict
from datetime import datetime

from torchvision.models import resnet18
from torch.utils.data.dataloader import DataLoader


# 当前脚本所在的绝对路径
BASE_DIR = os.path.dirname(__file__)
print(f"base dir is: {BASE_DIR}\n")
# 在系统路径中加入该脚本所在的上一级目录，便于查找项目模块的引用
sys.path.append(os.path.join(BASE_DIR, '..'))

from .trainer import Trainer
from Data.dataset import MyDataset
from visualization.plt import plot_line, show_conf_mat

from common_tools.logger import Logger
from common_tools.random_seed import setup_seed


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: is {device}\n")

# 设置随机种子，以便复现实验结果
setup_seed()


def parse_args():
    """从命令行解析参数，包括训练相关参数与数据目录"""

    parser = argparse.ArgumentParser(description='Training')
    # 最好为各参数都设置对应的类型(type)以及默认值(default)，同时添加上注释(help)
    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--bs', type=int, default=None, help='training batch size')
    parser.add_argument('--history', type=str, default=None, help='historical model weight')
    parser.add_argument('--weight', type=str, default=None, help='pre-trained model weight')
    parser.add_argument('--data_root_dir', type=str, default=None, help="path to your dataset")

    return parser.parse_args()


def reset_config(argts, cfg: EasyDict):
    """使用用户指定的配置参数更新默认配置"""

    update_dict = {
        'lr_init': argts.lr or cfg.lr_init,
        'train_bs': argts.bs or cfg.train_bs,
        'max_epoch': argts.max_epoch or cfg.max_epoch,
        'data_root_dir': argts.data_root_dir or cfg.data_root_dir,
        'pretrained_weight': argts.weight or cfg.pretrained_weight,
        'historical_weight': argts.history or cfg.historical_weight
    }
    # 让用户指定的参数覆盖默认配置
    cfg.update(update_dict)


if __name__ == '__main__':
    # step0: 配置项
    # 通常模型训练相关的配置会单独存放在一个脚本文件中
    # 配置参数可以存放在一个EasyDict()对象，它可以像访问类属性一样访问key-value对
    # 也就是：cfg.key = cfg[key]
    cfg = EasyDict()
    args = parse_args()
    reset_config(args, cfg)

    # 数据集目录
    train_dir = os.path.join(cfg.data_root_dir, 'train')
    valid_dir = os.path.join(cfg.data_root_dir, 'valid')
    # 最好检查下目录是否存在
    assert os.path.isdir(train_dir) and os.path.isdir(valid_dir)

    # 日志输出的目录
    now = datetime.now()
    # 月-日_时-分
    strftime = datetime.strftime(now, '%m-%d_%H-%M')
    # 利用时间作目录可以很方便地使得每次实验都有独立的日志目录
    log_dir = os.path.join(BASE_DIR, '..', 'results', strftime)
    os.makedirs(log_dir, exist_ok=True)

    # 使用自定义的Logger记录日志，可以很方便地将日志同时输出到控制台与文件
    log_path = os.path.join(log_dir, 'log.log')
    logger = Logger(log_path)

    # step1: 数据集
    # 构建Dataset和DataLoader实例
    # 数据增强可以设置在配置文件中
    train_data = MyDataset(train_dir, transform=cfg.transforms_train)
    valid_data = MyDataset(valid_dir, transform=cfg.transforms_valid)
    # 注意，为了使得训练集中每个batch大小一致，使用drop_last=True，从而令整体梯度更均匀
    # 训练集记得要打乱(shuffle=True)
    # num_workers推荐设置为cpu内核数一半，同时配合pin_memory=True，将数据提前加载至锁页内存，方便CUDA读取
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, shuffle=True,
                              num_workers=cfg.workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.valid_bs, num_workers=cfg.workers, pin_memory=True)

    # step2: 模型
    model = resnet18()
    # 加载预训练权重
    if os.path.exists(cfg.pretrained_weight):
        # 加载到cpu能节省主GPU卡的显存，同时兼容没有GPU的环境
        pretrained_state_dict = torch.load(cfg.pretrained_weight, map_location='cpu')
        model.load_state_dict(pretrained_state_dict)
        logger.info(f'load pre-trained weight {cfg.pretrained_weight}')

    # 通常图像分类项目，都需要将最后一层网络层的输出通道更改为类别数量
    # 此处可以更加实际情况更改
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, train_data.cls_num)

    # 加载之前训练好的权重
    if os.path.exists(cfg.historical_weight):
        state_dict = torch.load(cfg.historical_weight, map_location='cpu')
        # 这里假设模型权重是记录在checkpoint的model_state_dict这个key下
        # (因为有可能checkpoint不仅仅记录模型权重，还记录了学习率、训练的周期等信息)
        model.load_state_dict(state_dict['model_state_dict'])
        print(f"load historical weight {cfg.historical_weight}")

    # 将模型置于指定的设备上(cpu/gpu)
    model.to(device)

    # step3: 损失函数&优化器(根据实际情况更改)
    loss_func = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr_init, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    # 记录训练所采用的模型、损失函数、优化器以及配置参数，便于复现与分析实验
    logger.info(f"cfg:\n{cfg}\n loss_func:\n{loss_func}\n scheduler:\n{scheduler}\n "
                f"optimizer:\n{optimizer}\n model:\n{model}")

    # step4: 开始迭代训练
    best_acc = 0.
    best_epoch = 0
    loss_rec = {'train': [], 'valid': []}
    acc_rec = {'train': [], 'valid': []}

    start = datetime.now()
    for epoch in range(cfg.max_epoch):
        # 训练
        loss_train, acc_train, conf_mat_train, err_img_info_train = Trainer.train(
            train_loader, model, loss_func, optimizer,
            epoch, device, logger, cfg
        )
        # 验证
        loss_valid, acc_valid, conf_mat_valid, err_img_info_valid = Trainer.valid(
            valid_loader, model, loss_func, device
        )

        logger.info("Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} "
                    "Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
                        epoch + 1, cfg.max_epoch, acc_train, acc_valid,
                        loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

        # 根据学习率策略更新学习率
        scheduler.step()

        acc_rec['train'].append(acc_train)
        acc_rec['valid'].append(acc_valid)
        loss_rec['train'].append(loss_train)
        loss_rec['valid'].append(loss_valid)

        # 绘制混淆矩阵，将结果输出到文件
        show_conf_mat(conf_mat_train, tuple(range(train_data.cls_num)), 'train',
                      log_dir, epoch, verbose=epoch == cfg.max_epoch - 1)
        show_conf_mat(conf_mat_valid, tuple(range(valid_data.cls_num)), 'valid',
                      log_dir, epoch, verbose=epoch == cfg.max_epoch - 1)

        # 绘制loss和acc曲线，将结果输出到文件
        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec['train'], plt_x, loss_rec['valid'], 'Loss', log_dir)
        plot_line(plt_x, acc_rec['train'], plt_x, acc_rec['valid'], 'Acc', log_dir)

        # 当在验证集上取得更好的性能时 或者 是最后一个周期时，则保存模型参数
        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            if best_acc < acc_valid:
                best_epoch = epoch
                best_acc = acc_valid

            checkpoint = {
                'epoch': epoch,
                'best_acc': best_acc,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            pkl_name = 'checkpoint_best.pkl' if epoch != cfg.max_epoch - 1 else f'checkpoint_{epoch}.pkl'
            checkpoint_path = os.path.join(log_dir, pkl_name)
            torch.save(checkpoint, checkpoint_path)

            # 记录分类错误的图片信息
            err_img_info_file_name = 'err_imgs_best.pkl' if epoch != cfg.max_epoch - 1 else f'err_imgs_{epoch}.pkl'
            err_img_info_path = os.path.join(log_dir, err_img_info_file_name)
            err_info = {'train': err_img_info_train, 'valid': err_img_info_valid}
            with open(err_img_info_path, 'wb') as f:
                pickle.dump(err_info, f)

    end = datetime.now()
    time_used = (end - start).total_seconds()
    logger.info(f"Done on {end}, used {time_used}s, gain best acc {best_acc} in epoch {best_epoch}")
