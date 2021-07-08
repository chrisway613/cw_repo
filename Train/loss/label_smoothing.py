import torch
import torch.nn as nn


class LabelSmoothLoss(nn.Module):
    def __init__(self, smooth: float = 1e-3):
        super().__init__()
        self.smooth = smooth

    def forward(self, x: torch.Tensor, target: torch.Tensor):
        assert x.ndim - target.ndim == 1

        # weight = torch.ones_like(x)
        # weight *= self.smooth / (x.size(-1) - 1)
        # 先全部初始化为其它类别均分的平滑标签值
        weight = torch.full_like(x, self.smooth / (x.size(-1) - 1))

        truth_idx = target.unsqueeze(-1)
        # 然后填充真实类别对应的平滑标签值
        # 第2个参数的shape必须和源tensor一致，而且其值必须在源tensor对应维度的大小范围内
        weight.scatter_(-1, truth_idx, 1 - self.smooth)

        # 注意不是直接.mean()，而是在最后一个维度sum之后再求均值，代表所有样本的loss均值
        # 或者写成: .sum() / x.size(0)
        return -(weight * torch.log_softmax(x, -1)).sum(-1).mean()


if __name__ == '__main__':
    output = torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]])
    label = torch.tensor([2, 1, 1], dtype=torch.int64)

    criterion = LabelSmoothLoss(0.001)
    loss = criterion(output, label)

    print(f"CrossEntropy: {loss:.3f}")

