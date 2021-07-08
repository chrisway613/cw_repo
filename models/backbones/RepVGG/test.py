"""training or deploy mode test process"""

import os
import time
import argparse

import torch.backends
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

from .repvgg import get_RepVGG_func_by_name

from common_tools.evaluation import accuracy
from common_tools.random_seed import setup_seed
from common_tools.meters import AverageMeter, ProgressMeter


# 固定随机种子，以便复现结果
setup_seed()


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
    parser.add_argument('data', metavar='DIR', help='path to dataset')
    # 指示当前是训练还是部署
    parser.add_argument('mode', metavar='MODE', default='train', choices=['train', 'deploy'], help='train or deploy')
    parser.add_argument('weights', metavar='WEIGHTS', help='path to the weights file')
    # 指示使用哪种模型
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N',
                        help='mini-batch size (default: 100) for test')
    parser.add_argument('-r', '--resolution', default=224, type=int, metavar='R',
                        help='resolution (default: 224) for test')
    parser.add_argument('--display', type=int, default=10, help='log interval')

    return parser.parse_args()


def validate(loader, model, criterion, use_gpu, log_interval):
    # 占位6个字符，精确到小数点后3位
    batch_time = AverageMeter('Time', fmt=':6.3f')
    # 指数计数法，精确到小数点后4位
    losses = AverageMeter('Loss', fmt=':.4e')
    # 占位6个字符，精确到小数点后2位
    acc_top1 = AverageMeter('Acc@1', fmt=':6.2f')
    acc_top5 = AverageMeter('Acc@5', fmt=':6.2f')
    progress = ProgressMeter(
        len(loader),
        [batch_time, losses, acc_top1, acc_top5],
        prefix='Test: '
    )

    # switch to evaluation mode
    model.eval()

    # 在取消梯度的上下文环境中可节省内存占用
    with torch.no_grad():
        start = time.time()

        for i, (images, target) in enumerate(loader):
            if use_gpu:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)

            # feed forward
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), n=loader.batch_size)
            acc_top1.update(acc1[0], n=loader.batch_size)
            acc_top5.update(acc5[0], n=loader.batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start)
            start = time.time()

            if i % log_interval == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=acc_top1, top5=acc_top5))

    return getattr(acc_top1, 'avg')


if __name__ == '__main__':
    # i. configuration
    args = parse_args()

    # 基于ImageNet统计得到的图像均值与标准差
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalizer = transforms.Normalize(imagenet_mean, imagenet_std)

    if args.resolution == 224:
        trans = transforms.Compose([
            # 短边缩放到固定值，长边等比例缩放
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalizer
        ])
    else:
        trans = transforms.Compose([
            # 短边缩放到固定值，长边等比例缩放
            transforms.Resize(args.resolution),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalizer
        ])

    val_dir = os.path.join(args.data, 'val')

    # ii. load data
    dataset = datasets.ImageFolder(val_dir, transform=trans)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)

    # iii. build model
    model_build_func = get_RepVGG_func_by_name(args.arch)
    model = model_build_func(deploy=args.mode == 'deploy')

    # GPU/CPU
    if torch.cuda.is_available():
        model = model.cuda()
        use_gpu = True
    else:
        print("using CPU, this may be slow")
        use_gpu = False

    # 加载权重
    if os.path.isfile(args.weights):
        print(f"=> loading checkpoint '{args.weights}'")

        checkpoint = torch.load(args.weights)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']

        ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model.load_state_dict(ckpt)
    else:
        print(f"=> no checkpoint found at '{args.weights}'")

    # iv. define loss function(criterion), optimizer, lr scheduler
    criterion = nn.CrossEntropyLoss()
    if use_gpu:
        criterion = criterion.cuda()

    validate(loader, model, criterion, use_gpu, args.display)
