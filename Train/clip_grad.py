"""梯度裁剪，可基于梯度的值或范数进行裁剪"""

import torch


def clip_grad_value(params, min_value, max_value):
    """
        对梯度值进行截断
        :param params: 模型参数；
        :param min_value: 梯度截断的最小阀值
        :param max_value: 梯度截断最大阀值
        :return:
    """

    if isinstance(params, torch.Tensor):
        params = list(params)

    for p in filter(lambda param: param.grad is not None, params):
        # 若本身梯度的已经爆炸则先清零
        if torch.isnan(p.grad.data.sum()):
            p.grad.zero_()

        p.grad.data.clamp_(min=min_value, max=max_value)


def clip_grad_norm(params, max_norm, norm_type=2, eps=1e-6):
    """
        对梯度的范数进行截断
        :param params: 模型参数；
        :param max_norm: 范数截断阀值；
        :param norm_type: 范数形式
        :return: 截断后所有模型梯度的范数和
    """

    if max_norm <= 0:
        raise ValueError(f"'max_norm' should be positive, got {max_norm}")

    if isinstance(params, torch.Tensor):
        params = list(params)
    # 过滤掉不需要梯度的参数
    params = list(filter(lambda param: param.grad is not None, params))

    # 无穷范数就是所有参数的梯度绝对的最大值
    if norm_type == 'inf':
        total_norm = max(param.grad.data.abs().max() for param in params)
    else:
        total_norm = 0.
        for p in params:
            p_norm = p.grad.data.norm(norm_type)
            total_norm += p_norm.item() ** norm_type

        total_norm = total_norm ** (1. / norm_type)

    coef = max_norm / (total_norm + eps)
    if coef <= 0:
        raise RuntimeError(f"current coefficient {coef} is invalid!")
    if 0 < coef < 1.:
        for p in params:
            p.grad.data.mul_(coef)
