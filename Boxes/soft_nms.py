"""Soft-NMS with Pytorch Implementation."""

import torch

import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')


class SoftNms:
    def __init__(self, iou_thresh=.3, score_thresh=.5, method='linear', sigma2='.5'):
        """
        Soft-NMS实现，根据指定method对boxes的得分进行加权衰减
        :param iou_thresh: 与得分最高的box的IoU阀值；
        :param score_thresh: 最终要保留的boxes的得分阀值；
        :param method: 加权衰减得分的方法；
        :param sigma2: 当method是高斯时，计算公式中的sigma^2值
        """

        assert method in ('linear', 'gaussian', 'naive')
        self.method = method
        self.sigma2 = sigma2

        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

    def nms(self, boxes: torch.Tensor):
        """
        根据指定的权重(用于置信度)衰减方法实现soft nms
        :param boxes: (n,5), each is (score,x1,y1,x2,y2)
        :return: the remaining boxes after nms, and indices correspond to the original boxes tensor
        """

        # 先将boxes按置信度排序(由高到低)
        scores = boxes[:, 0]
        _, indices = scores.sort(descending=True)
        boxes = boxes[indices].float()

        for i, box in enumerate(boxes):
            if i == len(boxes) - 1:
                break

            # (1,4)
            best = box[1:].unsqueeze(0)
            # (m,4)
            others = boxes[i + 1:, 1:]

            # (m,2)
            xy_max = (torch.max(best[:, :2], others[:, :2]))
            # (m,2)
            xy_min = (torch.min(best[:, 2:], others[:, 2:]))
            # (m,)
            intersection = ((xy_min[:, 0] - xy_max[:, 0]) * (xy_min[:, 1] - xy_max[:, 1])).clamp_(min=0)

            # (1,)
            best_area = ((best[:, 2] - best[:, 0]) * (best[:, 3] - best[:, 1]))
            # (m,)
            others_area = ((others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1]))
            # (m,)
            union = (best_area + others_area - intersection).clamp_(min=0)

            # (m,)
            ious = intersection / (union + 1e-6)
            # (m,) True or False
            mask = ious > self.iou_thresh

            '''根据指定的方法加权衰减置信度(若是原版nms，则直接将置信度置0)'''
            if self.method == 'linear':
                weights = torch.ones_like(boxes[i + 1:, 0])
                weights[mask] -= ious[mask]
            elif self.method == 'gaussian':
                weights = torch.exp(- ious ** 2 / self.sigma2)
            else:
                weights = torch.ones_like(boxes[i + 1:, 0])
                weights[mask] = 0.

            boxes[i + 1:, 0] *= weights

        # 最终只保留置信度大于阀值的
        keep = boxes[:, 0] > self.score_thresh
        # indices[keep]代表保留下来的boxes对应到原来的哪些位置(索引)
        return boxes[keep], indices[keep]


if __name__ == '__main__':
    boxes = torch.tensor([
        [0.8, 10, 15, 80, 115],
        [0.9, 5, 20, 75, 105],
        [0.6, 30, 50, 200, 300],
        [0.75, 45, 60, 190, 288]
    ])

    fig = plt.figure()
    ax1 = plt.subplot(121)
    for box in boxes:
        box = box.numpy()
        xy = box[1:3] / 500
        w, h = (box[3:] - box[1:3]) / 500

        rect = plt.Rectangle(xy, w, h, fill=False, linewidth=1, edgecolor='b')
        ax1.add_patch(rect)

    module = SoftNms()
    kept_boxes, kept_indices = module.nms(boxes)
    print(f'After Soft Nms, the remaining boxes are:\n{kept_boxes}\n')
    print(f'indices are:\n{kept_indices}')

    ax2 = plt.subplot(122)
    for box in kept_boxes:
        box = box.float().numpy()
        xy = box[1:3] / 500
        w, h = (box[3:] - box[1:3]) / 500

        rect = plt.Rectangle(xy, w, h, fill=False, linewidth=1, edgecolor='r')
        ax2.add_patch(rect)

    plt.suptitle('Soft-NMS')
    plt.show()
