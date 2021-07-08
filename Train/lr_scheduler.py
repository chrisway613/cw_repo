"""自定义学习率策略"""

import torch

from packaging import version
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


PYTORCH_VERSION = version.parse(torch.__version__)


class LinearScheduler(_LRScheduler):
    """Linearly increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        assert num_iter > 1, f"`num_iter` must be larger than 1, got {num_iter}"

        self.end_lr = end_lr
        self.num_iter = num_iter

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        cur_iter = self.last_epoch
        if PYTORCH_VERSION < version.parse('1.1.0'):
            cur_iter += 1

        r = cur_iter / (self.num_iter - 1)
        # 这个列表的长度与模型参数量相等
        return [r * (self.end_lr - lr) + lr for lr in self.base_lrs]


class ExponentialScheduler(_LRScheduler):
    """Exponentially increases the learning rate between two boundaries over a number of iterations.

    Arguments:
        optimizer (torch.optim.Optimizer): wrapped optimizer.
        end_lr (float): the final learning rate.
        num_iter (int): the number of iterations over which the test occurs.
        last_epoch (int, optional): the index of last epoch. Default: -1.
    """

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1):
        assert num_iter > 1, f"`num_iter` must be larger than 1, got {num_iter}"

        self.end_lr = end_lr
        self.num_iter = num_iter

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # In earlier Pytorch versions last_epoch starts at -1, while in recent versions
        # it starts at 0. We need to adjust the math a bit to handle this. See
        # discussion at: https://github.com/davidtvs/pytorch-lr-finder/pull/42
        cur_iter = self.last_epoch
        if PYTORCH_VERSION < version.parse('1.1.0'):
            cur_iter += 1

        r = cur_iter / (self.num_iter - 1)
        # 这个列表的长度与模型参数量相等
        return [lr * (self.end_lr - lr) ** r for lr in self.base_lrs]
