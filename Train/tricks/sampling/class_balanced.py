"""各样本的采样权重与其所属类别的样本数量成反比，即：某类别样本数量越多，采样权重越小"""

import torch
import torchvision

from typing import Callable
from collections import defaultdict

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler


class ClassBalancedSampler(Sampler):
    def __init__(self, dataset: Dataset, indices: (list, torch.LongTensor) = None,
                 num_samples: int = None, callback_get_label: Callable = None):
        super().__init__(dataset)

        self.indices = range(len(dataset)) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        assert self.num_samples <= len(self.indices)

        # 根据数据集和样本索引获取样本标签的函数
        self.callback_get_label = callback_get_label

        label_counts = defaultdict(int)
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            label_counts[label] += 1

        # 各样本对应的采样权重：与其所属类别的样本总数成反比
        self.weights = torch.as_tensor(
            [1. / label_counts[self._get_label(dataset, idx)] for idx in self.indices], dtype=torch.double)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return iter([self.indices[i]
                     for i in torch.multinomial(self.weights, self.num_samples, replacement=True).tolist()])

    def _get_label(self, dataset: Dataset, index):
        """获取指定索引样本的标签"""

        if self.callback_get_label:
            return self.callback_get_label(dataset, index)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels[index].item()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return dataset.imgs[index][1]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[index][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[index][1]
        else:
            raise NotImplementedError
