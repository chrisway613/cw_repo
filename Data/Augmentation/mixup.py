import torch
import numpy as np


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""

    # 通过beta分布获得lambda，beta分布的参数alpha == beta，因此都是alpha
    # 返回的是标量(由于默认size=None)
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

    # 打乱张量的顺序，得到需要混叠的图片样本索引
    batch_size = x.size(0)
    # 默认返回的数据类型是torch.int64
    index = torch.randperm(batch_size, device=x.device)

    # mixup
    mixed_x = lam * x + (1 - lam) * x[index, :]
    # 注意，标签不需要混合计算，而是在后续计算loss的时候分别用这2个标签加权计算loss
    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # criterion是原生的loss
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    path_1 = r"sick.jpg"
    path_2 = r"puma.jpg"
    # path_1 = r"cw.jpg"
    # path_2 = r"titi.jpg"

    img_1 = cv2.imread(path_1)
    img_2 = cv2.imread(path_2)
    print(f"{img_1.shape} {img_2.shape}")
    img_1 = cv2.resize(img_1, (224,) * 2)
    img_2 = cv2.resize(img_2, (224,) * 2)

    alpha = 1.
    figsize = 15
    plt.figure(figsize=(figsize,) * 2)

    for i in range(1, 10):
        lam = np.random.beta(alpha, alpha)
        im_mixup = (img_1 * lam + img_2 * (1 - lam)).astype(np.uint8)
        im_mixup = cv2.cvtColor(im_mixup, cv2.COLOR_BGR2RGB)

        plt.subplot(3, 3, i)
        plt.title(f"lambda_{lam:.2f}")
        # 取消坐标轴显示
        plt.axis('off')
        # 或者用
        # plt.xticks([])
        # plt.yticks([])
        plt.imshow(im_mixup)

    # 调整子图之间在竖直方向上的间隔(同理wsapce调整水平方向间隔)
    plt.subplots_adjust(hspace=.3)
    plt.show()
