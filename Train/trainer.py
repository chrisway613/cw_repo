"""封装模型训练pipeline，包括在验证集上的评估"""

import logging
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist

from collections import Counter

from torch.optim.optimizer import Optimizer
from torch.utils.data.dataloader import DataLoader

from common_tools.meters import AverageMeter, ProgressMeter


class Trainer:
    @staticmethod
    def train(data_loader: DataLoader, model: nn.Module, loss_func: nn.Module, optimizer: Optimizer,
              epoch: int, device: torch.device, logger: logging.Logger, cfg):
        """训练pipeline, 此处以图像分类为例。
           cfg包含了模型训练相关的配置，注意，epoch从0开始。
        """

        # 注意先将模型设置为训练模式
        model.train()

        # 类别数量
        num_classes = data_loader.dataset.cls_num
        # 混淆矩阵 N*N 方阵
        conf_mat = np.zeros((num_classes,) * 2)

        # 样本标签
        sample_labels = []
        # 预测结果错误的图片信息(标签、预测类别、图片路径)，便于回溯
        err_img_info = []

        # 每个迭代计算下来的准确率和loss均值
        # acc_avg = loss_avg = 0.
        # 百分比形式，精确到小数点后2位
        acc = AverageMeter('Acc', fmt=':.2%')
        # 精确到小数点后4位
        losses = AverageMeter('Loss', fmt=':.4f')
        # 用于当前进度(准确率、损失、周期、批次)的显示
        prefix = f"[Rank{dist.get_rank()}]" if cfg.distributed else ""
        prefix += 'Training: Epoch[{:0>3}/{:0>3}] '.format(epoch, cfg.max_epoch)
        progress = ProgressMeter(len(data_loader), [acc, losses], prefix=prefix)

        for i, data in enumerate(data_loader):
            # 这里假设DataSet中封装的一条数为：图像、标签、图片路径
            inputs, labels, paths = data
            sample_labels.extend(labels.tolist())

            inputs, labels = inputs.to(device), labels.to(device)
            # 通常使用交叉熵损失要求标签是long类型，这部分可根据实际需求更改
            if labels.dtype != torch.long:
                labels = labels.long()

            # 先清空历史累计的梯度
            optimizer.zero_grad()

            # 前向反馈
            outputs = model(inputs)
            loss = loss_func(outputs, labels)

            # 后向传播
            loss.backward()
            optimizer.step()

            # 计算每个迭代(批次)的平均loss
            # loss_avg = (loss_avg * i + loss.item()) / (i + 1)
            losses.update(loss.item())

            _, preds = outputs.max(dim=1)
            for pred, label, path in zip(preds, labels, paths):
                pred = pred.item()
                label = label.item()
                conf_mat[label, pred] += 1.
                # 如果预测错误，则记录相关信息，后续便可以对此进行分析
                if pred != label:
                    err_img_info.append((label, pred, path))

            # acc_avg = conf_mat.trace() / conf_mat.sum() if conf_mat.sum() else 0.
            acc.update(conf_mat.trace() / conf_mat.sum() if conf_mat.sum() else 0.)

            # 到了该打印日志的迭代
            if i % cfg.log_interval == cfg.log_interval - 1:
                # logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".
                #             format(epoch + 1, cfg.max_epoch, i + 1, len(data_loader), loss_avg, acc_avg))
                log = progress.display(i, print_out=False)
                logger.info(log)

        # 释放缓存资源，使得其它GPU应用有更多可用资源(但不会增加Pytorch的可用资源)
        torch.cuda.empty_cache()
        logger.info(f"epoch:{epoch} sampler: {Counter(sample_labels)}")

        # return loss_avg, acc_avg, conf_mat, err_img_info
        return getattr(losses, 'avg'), getattr(acc, 'avg'), conf_mat, err_img_info

    @staticmethod
    def valid(data_loader: DataLoader, model: nn.Module, loss_func: nn.Module, device: torch.device):
        """在验证集上评估"""

        # 注意将模型切换为评估模式
        model.eval()

        # 类别数量
        num_classes = data_loader.dataset.cls_num
        # 混淆矩阵
        conf_mat = np.zeros((num_classes,) * 2)

        # loss_per_iter = []
        err_img_info = []

        acc = AverageMeter('Acc', fmt=':.2%')
        losses = AverageMeter('Loss', fmt=':.4f')

        # 可减少内存占用
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, labels, paths = data
                inputs, labels = inputs.to(device), labels.to(device)
                # 使用交叉熵等loss时要求标签是long类型的，这里可以根据实际情况更改
                if labels.dtype != torch.long:
                    labels = labels.long()

                outputs = model(inputs)
                loss = loss_func(outputs, labels)
                # loss_per_iter.append(loss.item())
                losses.update(loss.item())

                _, preds = outputs.max(dim=1)
                for pred, label, path in zip(preds, labels, paths):
                    pred = pred.item()
                    label = label.item()
                    conf_mat[label, pred] += 1.

                    # 记录预测错误样本的信息(真实类别、预测类别、图片路径)
                    if pred != label:
                        err_img_info.append((label, pred, path))

        # 释放缓存资源，使得其它GPU应用有更多可用资源(但不会增加Pytorch的可用资源)
        torch.cuda.empty_cache()
        # 准确率=预测正确的样本数/所有样本数
        # acc_avg = conf_mat.trace() / conf_mat.sum()
        acc.update(conf_mat.trace() / conf_mat.sum() if conf_mat.sum() else 0.)

        # return np.mean(loss_per_iter), acc_avg, conf_mat, err_img_info
        return getattr(losses, 'avg'), getattr(acc, 'avg'), conf_mat, err_img_info
