"""Pytorch分布式训练pipeline 脚本调用方式：
       python -m torch.distributed.launch --nproc_per_node [num_procs] train_distributed.py [Other parameters]
   若想指定仅使用部分显卡，可以在 python -m 前加上：[CUDA_VISIBLE_DEVICES]=0,1,2,.."""


import os
import sys
import pickle
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from datetime import datetime
from contextlib import contextmanager

from torchvision.models import resnet18
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# 当前脚本所在的绝对路径
BASE_DIR = os.path.dirname(__file__)
# 在系统路径中加入该脚本所在的上一级目录，便于查找项目模块的引用
sys.path.append(os.path.join(BASE_DIR, '..'))

from common_tools.logger import Logger
from common_tools.random_seed import setup_seed

from trainer import Trainer
from Data.dataset import MyDataset
from configs.flower102_config import cfg
from visualization.plt import plot_line, show_conf_mat


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')

    parser.add_argument('--max_epoch', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--bs', type=int, default=None, help='training batch size')
    parser.add_argument('--history', type=str, default=None, help='historical model weight')
    parser.add_argument('--weight', type=str, default=None, help='pre-trained model weight')
    parser.add_argument('--data_root_dir', type=str, default=None, help="path to your dataset")
    parser.add_argument('--mixup', action='store_true', help='whether to use mixup')
    parser.add_argument('--label_smooth', action='store_true', help='whether to use label-smooth')
    # action='store_true'代表：若在命令行指定了该参数，则将参数值设置为True，否则为False
    parser.add_argument('--cb', action='store_true', help='whether to use class-balanced sampling')
    parser.add_argument('--pb', action='store_true', help='whether to use progressively-balanced sampling')
    parser.add_argument('--ohem', action='store_true', help='whether to use online hard example mining')
    parser.add_argument('--use_sync_bn', action='store_true', help='whether to synchronize bn')
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args()


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training"""

    if not dist.is_available():
        return

    if not dist.is_initialized():
        return

    if not dist.get_world_size() > 1:
        return

    dist.barrier()


def gather(data, device):
    """将各进程的数据(不一定是tensor类型也不要求同样大小)收集起来并且同步到各个进程"""

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.IntTensor([tensor.numel()]).to(device)
    size_list = [torch.IntTensor([0]).to(device) for _ in range(num_gpus)]
    dist.all_gather(size_list, local_size)

    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)
    tensor_list = [torch.ByteTensor(size=(max_size,)).to(device) for _ in size_list]

    if local_size != max_size:
        # we pad the tensor because torch all_gather does not support
        # gathering tensors of different shapes
        # padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        padding = torch.ByteTensor(size=(max_size - local_size,)).to(device)
        tensor = torch.cat((tensor, padding), dim=0)

    # receiving Tensor from all ranks
    dist.all_gather(tensor_list, tensor)

    return tensor_list, size_list


def is_pytorch_1_1_0_or_later():
    return [int(_) for _ in torch.__version__.split(".")[:3]] >= [1, 1, 0]


def reset_config(argts):
    """使用用户指定的配置参数更新默认配置"""

    update_dict = {
        'lr': argts.lr or cfg.lr,
        'train_bs': argts.bs or cfg.train_bs,
        'max_epoch': argts.max_epoch or cfg.max_epoch,
        'data_root_dir': argts.data_root_dir or cfg.data_root_dir,
        'pretrained_weight': argts.weight or cfg.pretrained_weight,
        'historical_weight': argts.history or cfg.historical_weight,
        'cb': argts.cb or cfg.cb,
        'pb': argts.pb or cfg.pb,
        'mixup': argts.mixup or cfg.mixup,
        'label_smooth': argts.label_smooth or cfg.label_smooth,
        'ohem': argts.ohem or cfg.ohem,
        'use_sync_bn': argts.use_sync_bn or cfg.use_sync_bn
    }
    cfg.update(update_dict)


def check_data_dir(*path):
    for p in path:
        assert os.path.exists(p), f"got invalid path: {os.path.abspath(p)}\n"


@contextmanager
def distributed_master_first(rank: int):
    """分布式模式下，主进程优先执行上下文管理器下的操作，待主进程执行完，其余进程才得以开始执行
       用法：
           with distributed_master_first:
               do something
    """

    # rank > 1代表副进程
    # 在此先等待(同步)
    if rank:
        synchronize()

    # 相当于一个return
    yield
    # yield后面的语句待退出上下文环境时再执行
    # rank=0代表主进程
    # 此时主进程已执行完上下文环境中的语句
    if not rank:
        # 主进程通知其余进程(同时也有等待其余进程的效果)
        synchronize()


if __name__ == '__main__':
    # step0: 日志输出设置、命令行参数解析、训练相关配置项

    now = datetime.now().strftime('%m-%d_%H-%M')
    out_dir = os.path.join(BASE_DIR, '..', 'train_results', now)
    log_path = os.path.join(out_dir, f"train.log")
    logger = Logger(log_path)
    logger.info(f"[Dir] {BASE_DIR}")

    num_gpus = int(os.environ.get('WORLD_SIZE', 1))

    args = parse_args()
    args.distributed = num_gpus > 1
    if args.distributed:
        logger.info(f"[Local Rank] {args.local_rank}")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group('nccl')
        # 同步 待所有进程都完成进程组相关的初始化之后再开始执行下一步
        # 避免由于某进程未加入进程组从而失联
        synchronize()
    logger.info(f"[Train Distributed] {args.distributed}")

    # 设置随机种子，以便复现实验结果
    # 注意，分布式模式下各进程的随机种子需要不同，以免在使用一些概率性数据增强时造成数据同态
    # 数据同态性会降低数据质量
    seed = 0 if not args.distributed else 1 + dist.get_rank()
    setup_seed(seed)

    reset_config(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("[Device] " + f"gpu:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")

    # step1: 数据集

    # 数据集目录
    train_dir = os.path.join(cfg.data_root_dir, 'train')
    valid_dir = os.path.join(cfg.data_root_dir, 'valid')
    # 检查目录是否存在
    check_data_dir(train_dir, valid_dir)

    # 构建Dataset和DataLoader实例
    train_data = MyDataset(train_dir, transform=cfg.transforms_train)
    valid_data = MyDataset(valid_dir, transform=cfg.transforms_valid)

    if args.distributed:
        loader_shuffle = False
        # 默认shuffle=True
        train_sampler = DistributedSampler(train_data)
        valid_sampler = DistributedSampler(valid_data, shuffle=False)
    else:
        loader_shuffle = True
        train_sampler = valid_sampler = None

    # 注意，为了使得训练集中每个batch大小一致，使用drop_last=True，从而令整体梯度更均匀
    # num_workers推荐设置为cpu内核数一半，同时配合pin_memory=True，将数据提前加载至锁页内存，方便CUDA读取
    train_loader = DataLoader(train_data, batch_size=cfg.train_bs, sampler=train_sampler,
                              shuffle=loader_shuffle, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_data, batch_size=cfg.valid_bs, sampler=valid_sampler,
                              num_workers=cfg.num_workers, pin_memory=True)

    # step2: 模型

    model = resnet18()
    # 加载预训练权重(主进程加载即可)
    if os.path.exists(cfg.pretrained_weight) and dist.get_rank() == 0:
        # 加载到cpu能节省主GPU卡的显存，同时兼容没有GPU的环境
        pretrained_state_dict = torch.load(cfg.pretrained_weight, map_location='cpu')
        model.load_state_dict(pretrained_state_dict)
        logger.info(f'load pre-trained weight from: {cfg.pretrained_weight}')

    # 通常图像分类项目，都需要将最后一层网络层的输出通道更改为类别数量
    # 此处可以更加实际情况更改
    num_feats = model.fc.in_features
    model.fc = nn.Linear(num_feats, train_data.cls_num)

    # 将BN转换成多卡同步BN
    if args.distributed and cfg.use_sync_bn:
        # 注意，sync bn转换支持单进程单卡模式，且需要在torch.distributed.init_process_group()之后调用
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # 加载之前训练好的权重(主进程加载即可)
    # 注意顺序：先加载权重，再放置到对应的设备(device)，最后用DDP封装
    if os.path.exists(cfg.historical_weight) and dist.get_rank() == 0:
        state_dict = torch.load(cfg.historical_weight, map_location='cpu')
        # 这里假设模型权重是记录在checkpoint的model这个key下
        # (因为有可能checkpoint不仅仅记录模型权重，还记录了学习率、训练的周期等信息)
        model.load_state_dict(state_dict['model'])
        print(f"load historical weight {cfg.historical_weight}")

    model.to(device)
    if args.distributed:
        model = DDP(model, device_ids=[args.local_rank])

    # step3: 损失函数&优化器(根据实际情况更改)
    # 注意，分布式训练下，先用DDP封装了再设置优化器，
    # 这样能保证各进程模型初始化一致，优化器设置一致，再加上梯度一致(平均)，从而各模型最终的参数一致
    optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)
    if os.path.exists(cfg.historical_weight):
        state_dict = torch.load(cfg.historical_weight, map_location='cpu')
        optimizer.load_state_dict(state_dict['optimizer'])
        scheduler.load_state_dict(state_dict['scheduler'])

    loss_func = nn.CrossEntropyLoss().to(device)

    # 记录训练所采用的模型、损失函数、优化器以及配置参数，便于复现与分析实验
    logger.info(f"[Configuration]\n{cfg}\n\n[Loss Function]\n{loss_func}\n\n [LR Scheduler]\n{scheduler}\n\n"
                f"[Optimizer]\n{optimizer}\n\n [Model]\n{model}")

    # step4: 迭代训练
    best_acc = 0.
    best_epoch = 0
    loss_rec = {'train': [], 'valid': []}
    acc_rec = {'train': [], 'valid': []}

    start = datetime.now()
    for epoch in range(cfg.max_epoch):
        if args.distributed:
            # DistributedSampler中shuffle的随机种子就是epoch，
            # 因此，每个epoch调用该方法代表在每个epoch都有不同的shuffle效果
            # 但各个进程之间是同样的shuffle效果，然后取出它们对应的数据划分部分。
            train_loader.sampler.set_epoch(epoch)
            valid_loader.sampler.set_epoch(epoch)

        # 1.1.0之前的版本在迭代训练前要先对学习率策略做一次step
        if not is_pytorch_1_1_0_or_later():
            scheduler.step()

        # 训练
        loss_train, acc_train, conf_mat_train, err_img_info_train = Trainer.train(
            train_loader, model, loss_func, optimizer,
            epoch, device, logger, cfg
        )
        # 验证
        loss_valid, acc_valid, conf_mat_valid, err_img_info_valid = Trainer.valid(
            valid_loader, model, loss_func, device
        )

        info = f"Rank{dist.get_rank()}:" if args.distributed else ""
        info += "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} " \
                "Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
                     epoch + 1, cfg.max_epoch, acc_train, acc_valid,
                     loss_train, loss_valid, optimizer.param_groups[0]["lr"])
        logger.info(info)

        if is_pytorch_1_1_0_or_later():
            # 根据学习率策略更新学习率
            scheduler.step()

        if args.distributed:
            acc_train = torch.tensor(acc_train, device=device)
            acc_valid = torch.tensor(acc_valid, device=device)
            loss_train = torch.tensor(loss_train, device=device)
            loss_valid = torch.tensor(loss_valid, device=device)

            conf_mat_train = torch.from_numpy(conf_mat_train).to(device)
            conf_mat_valid = torch.from_numpy(conf_mat_valid).to(device)

            dist.reduce(acc_train, 0)
            dist.reduce(acc_valid, 0)
            dist.reduce(loss_train, 0)
            dist.reduce(loss_valid, 0)

            dist.reduce(conf_mat_train, 0)
            dist.reduce(conf_mat_valid, 0)

            tensor_list_err_img_info_train, size_list_err_img_info_train = gather(err_img_info_train, device)
            tensor_list_err_img_info_valid, size_list_err_img_info_valid = gather(err_img_info_valid, device)

            acc_train = acc_train.item() / num_gpus
            acc_valid = acc_valid.item() / num_gpus
            loss_train = loss_train.item() / num_gpus
            loss_valid = loss_valid.item() / num_gpus

            conf_mat_train = conf_mat_train.cpu().numpy()
            conf_mat_valid = conf_mat_valid.cpu().numpy()

            if dist.get_rank() == 0:
                err_img_info_train = []
                # 依次取出各个进程的数据
                for size, tensor in zip(size_list_err_img_info_train, tensor_list_err_img_info_train):
                    # 去除pad的部分(由于gather的时候要求各进程的数据大小一致，因此进行了pad)，仅获取该进程本身的数据量
                    buffer = tensor.cpu().numpy().tobytes()[:size]
                    err_img_info_train += pickle.loads(buffer)

                err_img_info_valid = []
                # 依次取出各个进程的数据
                for size, tensor in zip(size_list_err_img_info_valid, tensor_list_err_img_info_valid):
                    # 去除pad的部分(由于gather的时候要求各进程的数据大小一致，因此进行了pad)，仅获取该进程本身的数据量
                    buffer = tensor.cpu().numpy().tobytes()[:size]
                    err_img_info_valid += pickle.loads(buffer)

        acc_rec['train'].append(acc_train)
        acc_rec['valid'].append(acc_valid)
        loss_rec['train'].append(loss_train)
        loss_rec['valid'].append(loss_valid)

        # 分布式训练下只有主进程进入该条件分支
        if not args.distributed or dist.get_rank() == 0:
            # 绘制混淆矩阵，将结果输出到文件
            show_conf_mat(conf_mat_train, tuple(range(train_data.cls_num)), 'train',
                          out_dir, epoch, verbose=epoch == cfg.max_epoch - 1)
            show_conf_mat(conf_mat_valid, tuple(range(valid_data.cls_num)), 'valid',
                          out_dir, epoch, verbose=epoch == cfg.max_epoch - 1)

            # 绘制loss和acc曲线，将结果输出到文件
            plt_x = np.arange(1, epoch + 2)
            plot_line(plt_x, loss_rec['train'], plt_x, loss_rec['valid'], 'Loss', out_dir)
            plot_line(plt_x, acc_rec['train'], plt_x, acc_rec['valid'], 'Acc', out_dir)

            # 记录分类错误的图片信息
            err_img_info_file_name = 'err_imgs_best.pkl' if epoch != cfg.max_epoch - 1 else f'err_imgs_{epoch}.pkl'
            err_img_info_path = os.path.join(out_dir, err_img_info_file_name)
            err_info = {'train': err_img_info_train, 'valid': err_img_info_valid}
            with open(err_img_info_path, 'wb') as f:
                pickle.dump(err_info, f)

            # 当在验证集上取得更好的性能时 或者 是最后一个周期时，则保存模型参数
            if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
                if best_acc < acc_valid:
                    best_epoch = epoch
                    best_acc = acc_valid

                checkpoint = {
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model': model.module.state_dict() if args.distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }

                pkl_name = 'checkpoint_best.pkl' if epoch != cfg.max_epoch - 1 else f'checkpoint_{epoch}.pkl'
                checkpoint_path = os.path.join(out_dir, pkl_name)
                torch.save(checkpoint, checkpoint_path)

    end = datetime.now()
    time_used = (end - start).total_seconds()
    if not args.distributed or dist.get_rank() == 0:
        logger.info(f"Done on {end}, used {time_used}s, gain best acc {best_acc} in epoch {best_epoch}")
