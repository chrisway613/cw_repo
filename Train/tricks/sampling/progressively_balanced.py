"""Progressively-Balanced Sampling 渐进式采样策略
   各类别样本的采样权重随着Epoch的增加渐进式地发生变化，使得原始分布逐渐变为类别均衡的分布"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data.sampler import WeightedRandomSampler


class PBSamplingGenerator:
    def __init__(self, T: int, train_targets: (list, tuple, torch.LongTensor)):
        # 最大训练周期
        self.T = T
        # 训练样本标签
        self.train_targets = [int(label) for label in train_targets]
        # 各类别样本数量
        self.num_per_cls = [cnt for cnt in np.bincount(self.train_targets) if cnt]
        assert len(self.num_per_cls) == len(np.unique(self.train_targets))

        # ib: instance-balanced 实例均衡 即采样权重与类别样本数成正比，类别样本数越多，其下的样本被抽取到的概率越大
        self.ib = self._compute_cls_prob(1)
        # cb: class-balanced 类别均衡 即采样权重与类别无关，各个类别的样本被抽取的概率相同
        self.cb = self._compute_cls_prob(0)

    def _compute_cls_prob(self, q):
        """计算各类别样本被采样的概率，q越接近0则越倾向于类别均衡；越接近1越倾向于实例均衡"""

        num_pow = [pow(num, q) for num in self.num_per_cls]
        num_pow_sum = sum(num_pow)

        return [p / num_pow_sum for p in num_pow]

    def __call__(self, t: int):
        """计算当前训练迭代下的渐进式采样概率"""

        # 随着训练的进行，采样概率会越来越倾向于类别均衡
        pb_prob = (1 - t / self.T) * np.array(self.ib) + t / self.T * np.array(self.cb)
        # 仿照WeightedRandomSampler的实现，dtype是double类型
        pb_prob = torch.as_tensor(pb_prob, dtype=torch.double)

        # 各个训练样本依据它们的类别获得对应的采样权重
        weights = pb_prob[self.train_targets]
        sampler = WeightedRandomSampler(weights, len(weights))

        return sampler, pb_prob

    def plot_prob_t(self, t):
        _, prob = self(t)
        x = range(len(prob))

        plt.plot(x, prob)
        plt.title(f'Progressively-Balanced t={t},T={self.T}')
        plt.xlabel('class index')
        plt.ylabel('sample probability')
        plt.show()
