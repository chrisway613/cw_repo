# Online Hard Negative Mining
# 这种实现方式并非仅选取loss最大的样本进行反向传播，
# 而是选取了loss最小和loss最大的一批样本，而loss处于中间水平的样本被忽略

import torch
import torch.nn.functional as F

from common_tools.log_sum_exp import log_sum_exp


if __name__ == '__main__':
    # batch size=6，其中每个实例有8个样本，预测3个类别
    num_classes = 3
    # (6,8,3)
    pred = torch.randn(6, 8, num_classes)
    # (6,8)
    target = torch.randint(0, num_classes, (6, 8)).long()
    # 先计算出所有样本的loss
    # (6,8,1)->(6,8)
    loss = (log_sum_exp(pred) - pred.gather(dim=-1, index=target.unsqueeze(dim=-1))).squeeze()
    print(f'loss:\n{loss}\n')
    # 这里为了方便，定义loss小于均值的一半的那批样本为正样本
    # (6,8) True or False
    pos = loss < .5 * loss.mean(dim=-1, keepdim=True)
    print(f'positive indices:\n{pos}\n')
    # 负样本数量相对于正样本数量的比例
    neg_ratio = 3.

    # filter out positive boxes
    loss[pos] = 0.
    # 将样本根据loss由大到小排序（注意，正样本的loss已置为0）
    # (6,8)
    _, loss_idx = loss.sort(dim=-1, descending=True)
    # idx_rank代表各个样本对应的loss是第几大的（0代表最大）
    # (6,8)
    _, idx_rank = loss_idx.sort(dim=-1)
    # 每个实例的正样本数量
    # (6,1)
    num_pos = pos.long().sum(dim=-1, keepdim=True)
    # 正样本至少占1个，因此负样本数量最多是样本总数减去1
    # (6,1)
    num_neg = torch.clamp(neg_ratio * num_pos, max=loss.size(-1) - 1)
    # (6,8) True or False
    neg = idx_rank < num_neg.expand_as(idx_rank)
    print(f'negative indices:\n{neg}\n')

    # Confidence Loss Including Positive and Negative Examples
    '''取出采样后的预测结果与标签'''
    # (n_sampled,)
    # target = target[(pos + neg).gt(0)]
    target = target[pos | neg]
    # (6,8) -> (6,8,3)
    pos_idx = pos.unsqueeze(dim=-1).expand_as(pred)
    # (6,8) -> (6,8,3)
    neg_idx = neg.unsqueeze(dim=-1).expand_as(pred)
    # (n_sampled*n_classes) -> (n_sampled,n_classed)
    # pred = pred[(pos_idx + neg_idx).gt(0)].view(-1, num_classes)
    pred = pred[pos_idx | neg_idx].view(-1, num_classes)

    # 计算采样样本的loss
    loss = F.cross_entropy(pred, target)
    print(f'final loss:\n{loss}')
