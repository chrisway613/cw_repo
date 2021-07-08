import torch


def log_sum_exp(x):
    x_max, _ = x.max(dim=-1, keepdim=True)
    # log-softmax取反，返回的这个结果减去x对应通道的结果即可作为交叉熵损失
    return x_max + torch.log(torch.exp(x - x_max).sum(dim=-1, keepdim=True))


if __name__ == '__main__':
    prob = torch.randn(2, 3)
    print(prob)
    # 取第一个类别的概率计算损失
    loss = log_sum_exp(prob) - prob[:, :1]
    print(loss)
