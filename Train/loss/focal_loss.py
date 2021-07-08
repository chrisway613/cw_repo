import torch
import torch.nn as nn


# class FocalLoss(nn.BCEWithLogitsLoss):
#     """Focal Loss, 针对困难样本及样本不均衡问题。
#        这里继承BCEWithLogitsLoss，其结合了BCE和Sigmoid，这样输入的预测结果就无需先经过sigmoid，可以直接是网络输出的logits；
#        另外，BCEWithLogitsLoss还使用了log-sum-exp，更稳定。"""
#
#     def __init__(self, alpha: float = .25, gamma: float = 2., reduction: str = 'mean'):
#         super().__init__(reduction=reduction)
#
#         self.alpha = alpha
#         self.gamma = gamma
#
#     def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         bs = len(targets)
#         assert len(inputs) == bs
#
#         # 注意，以下detach()将张量从计算图中剥离，不涉及梯度，
#         # 这是因为weight在BCEWithLogitsLoss中是注册到buffer的，不应该存在梯度，否则会抛出异常
#         targets_onehot = targets.detach()
#         if inputs.shape != targets.shape:
#             targets_onehot = torch.zeros_like(inputs, device=inputs.device).detach()
#             for i in range(bs):
#                 cls_idx = targets[i].item()
#                 targets_onehot[i, cls_idx] = 1.
#
#         inputs_sigmoid = torch.sigmoid(inputs).detach()
#         # 权重需要在前向过程中根据预测结果和标签进行设置，
#         # 因为每个batch的数据量可能不同(比如last batch通常较小，又或者训练集和验证集bs不同)
#         weight = self.alpha * targets_onehot * (1 - inputs_sigmoid) ** self.gamma + \
#             (1 - self.alpha) * (1 - targets_onehot) * inputs_sigmoid ** self.gamma
#         # 权重不属于模型的参数，因此注册到buffer(在buffer中就可以被state_dict记录下来)
#         self.register_buffer('weight', weight)
#
#         return super().forward(inputs, targets_onehot)


class FocalLoss(nn.BCEWithLogitsLoss):
    """Focal Loss, 针对困难样本及样本不均衡问题。
       这里继承BCEWithLogitsLoss，其结合了BCE和Sigmoid，这样输入的预测结果就无需先经过sigmoid，可以直接是网络输出的logits；
       BCEWithLogitsLoss使用了log-sum-exp，更稳定。
       另外，BCE对模型输出的logits使用sigmoid，相比于使用softmax，其不会在类别之间形成竞争关系，还可以支持多类别标签，
       即：一个样本可同时属于多个类别，可兼容性更强。"""

    def __init__(self, alpha: float = .25, gamma: float = 2., reduction: str = 'mean'):
        super().__init__(reduction=reduction)

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if inputs.shape != targets.shape:
            # inputs需要是(B,C,*), targets是(B,*)
            # 并且0 ≤ targets[i] ≤ C−1，0 <= i < B
            assert inputs.ndim - targets.ndim == 1
            bs, num_cls = inputs.shape[:2]
            assert len(targets) == bs and inputs.shape[2:] == targets.shape[1:]
            assert 0 <= targets.min() <= targets.max() <= num_cls - 1

            # 注意，以下detach()将张量从计算图中剥离，不涉及梯度，
            # 这是因为weight在BCEWithLogitsLoss中是注册到buffer的，不应该存在梯度，否则会抛出异常
            # (B,C,H,W)
            targets_onehot = torch.zeros_like(inputs, device=inputs.device).detach()
            targets_onehot.scatter_(1, targets.unsqueeze(1), 1.)
        # 这种情况下标签也不一定是one_hot，可以是多类别标签，即一个样本同时属于多个类别
        else:
            # 注意要转换成float，因为BCE Loss的接口要求标签是float类型
            targets_onehot = targets.float().detach()

        # 注意 这里经过sigmoid只是为了计算focal loss权重，而输入到bce loss接口时应该传入模型输出的logits
        prob = torch.sigmoid(inputs).detach()
        # 权重需要在前向过程中根据预测结果和标签进行设置，因为每个batch的数据量可能不同(比如last batch通常较小，又或者训练集和验证集bs不同)
        # (B,C,H,W)
        weight = self.alpha * targets_onehot * (1 - prob) ** self.gamma + \
            (1 - self.alpha) * (1 - targets_onehot) * prob ** self.gamma
        # 权重不属于模型的参数，因此注册到buffer(在buffer中就可以被state_dict记录下来)
        self.register_buffer('weight', weight)

        return super().forward(inputs, targets_onehot)


if __name__ == '__main__':
    # 2个样本(每个size 4x4)，3个类别
    x = torch.randn(2, 3, 4, 4)
    # 每个样本仅属于1个类别
    # y = torch.randint(0, 3, (2, 4, 4))
    # 多类别标签，每个样本可同时属于多个类别
    y = torch.randint(0, 2, (2, 3, 4, 4))

    loss_func = FocalLoss(reduction='none')
    loss = loss_func(x, y)
    print(loss)
    print('-' * 60, '\n')

    '''calculation by self-definition'''

    # 单类别标签下的one_hot转换
    # y_one_hot = torch.zeros_like(x)
    # y_one_hot.scatter_(1, y.unsqueeze(1), 1.)
    # 多类别标签直接由原标签转换成float
    y_one_hot = y.float()
    print(y_one_hot)
    print('-' * 60, )

    alpha = .25
    gamma = 2.
    x_sigmoid = torch.sigmoid(x)
    print(x_sigmoid)
    print('-' * 60, )

    weights = alpha * y_one_hot * (1 - x_sigmoid) ** gamma + (1 - alpha) * (1 - y_one_hot) * x_sigmoid ** gamma
    print(weights)
    print('-' * 60, )

    bce_loss = -y_one_hot * torch.log(x_sigmoid) - (1 - y_one_hot) * torch.log(1 - x_sigmoid)
    focal_loss = weights * bce_loss
    print(focal_loss)

    assert torch.allclose(loss, focal_loss)
