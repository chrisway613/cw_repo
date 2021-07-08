"""Pytorch Implementation of Weighted Boxes Fusion, details see:
   https://arxiv.org/abs/1910.13302"""

import torch
import matplotlib
import matplotlib.pyplot as plt


matplotlib.use('TkAgg')


class BoxFusion:
    def __init__(self, iou_thresh=.55, score_thresh=0., weights=1., conf_type='avg', allows_overflow=False):
        """
        对多个模型预测的bboxes进行加权融合，而非像NMS般去掉某些bboxes.
        :param iou_thresh: IoU阀值，超过该阀值则认为2个bboxes匹配；
        :param score_thresh: 忽略(舍弃)低于该置信度的bboxes；
        :param weights: 各个模型的权重；
        :param conf_type: 加权融合计算置信度的方式；
        :param allows_overflow: 是否允许置信度大于1
        """

        assert 0 < iou_thresh < 1, 0 <= score_thresh < 1
        self.iou_thresh = iou_thresh
        self.score_thresh = score_thresh

        if isinstance(weights, (int, float)):
            weights = [weights]
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float)
        self.weights = weights

        assert conf_type in ('avg', 'max', 'box_and_model_avg', 'absent_model_aware_avg')
        self.conf_type = conf_type
        self.allows_overflow = allows_overflow

    @staticmethod
    def calculate_match_ious(boxes, box):
        """计算box与一组boxes的最大iou，以及最匹配的是哪个"""
        xy_max = torch.max(boxes[:, :2], box[None, :2])
        xy_min = torch.min(boxes[:, 2:], box[None, 2:])
        wh_inter = xy_min - xy_max
        intersection = (wh_inter[:, 0] * wh_inter[:, 1]).float().clamp_(min=0.)

        whs = boxes[:, 2:] - boxes[:, :2]
        wh = box[None, 2:] - box[None, :2]
        union = (whs[:, 0] * whs[:, 1] + wh[:, 0] * wh[:, 1]).float().clamp_(min=0.)

        ious = intersection / (union - intersection + 1e-6)
        # max iou, index
        return torch.max(ious, 0)

    @classmethod
    def calculate_weighted_box(cls, boxes):
        """基于一组bboxes计算加权融合的bbox"""

        weighted_box = torch.empty(8, dtype=boxes[0].dtype, device=boxes[0].device)

        boxes = torch.stack(boxes)
        if len(boxes):
            label = boxes[0][0]
            conf_mean = boxes[:, 1].mean()
            conf_sum = boxes[:, 1].sum()
            model_weight_sum = boxes[:, 2].sum()

            locs = boxes[:, 4:]
            confs = boxes[:, 1].unsqueeze(dim=1)
            weighted_locs = (confs * locs).sum(dim=0)

            weighted_box[0] = label
            weighted_box[1] = conf_mean
            weighted_box[2] = model_weight_sum
            # model index field is retained for consistency, but will not be used.
            weighted_box[3] = -1
            weighted_box[4:] = weighted_locs / conf_sum

        # [label,mean_score,model_weight_sum,model_index,x1,y1,x2,y2]
        return weighted_box

    def filter_boxes(self, boxes, scores, labels):
        """过滤掉置信度太低以及坐标不合法的bboxes"""

        filtered_boxes = {}
        for model_idx, (model_pred_boxes, model_pred_scores, model_pred_labels) in enumerate(zip(boxes, scores, labels)):
            assert model_idx < len(self.weights)
            assert len(model_pred_boxes) == len(model_pred_scores) == len(model_pred_labels)

            for box, score, label in zip(model_pred_boxes, model_pred_scores, model_pred_labels):
                # 忽略置信度低于阀值的bbox
                if score < self.score_thresh:
                    continue

                box.clamp_(min=0)
                # 忽略掉坐标不合法的bbox
                if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
                    continue

                model_weight = self.weights[model_idx]
                prefix = torch.tensor([label, score * model_weight, model_weight, model_idx], device=box.device)
                # (8,)
                new_box = torch.cat([prefix, box.float()])

                c = int(label)
                filtered_boxes.setdefault(c, [])
                filtered_boxes[c].append(new_box)

        for c, boxes_list in filtered_boxes.items():
            filtered_boxes[c] = torch.stack(boxes_list)

        # class->boxes(n,8)
        return filtered_boxes

    def box_matching(self, weighted_boxes, box):
        assert isinstance(weighted_boxes, (list, torch.Tensor))
        if not len(weighted_boxes):
            return -1, self.iou_thresh

        if isinstance(weighted_boxes, list):
            weighted_boxes = torch.stack(weighted_boxes)

        matched_idx = -1
        matched_iou = self.iou_thresh

        # 过滤，仅保留相同类别的bboxes
        same_cls_indices = weighted_boxes[:, 0] == box[0]
        same_cls_weighted_boxes = weighted_boxes[same_cls_indices]
        if len(same_cls_weighted_boxes):
            iou, idx = self.calculate_match_ious(same_cls_weighted_boxes[:, 4:], box[4:])
            if iou > matched_iou:
                matched_iou = iou
                matched_idx = idx

        return matched_idx, matched_iou

    def get_fusion_boxes(self, raw_boxes, scores, labels):
        """
        对相互重叠度高的bboxes进行加权融合
        :param raw_boxes: (n_models, n_boxes_pred_by_each_model, 4) 各模型预测的bboxes；
        :param scores: (n_models, n_boxes_pred_by_each_model) 各bboxes对应的置信度；
        :param labels: (n_models, n_boxes_pred_by_each_model) 各bboxes对应的标签(物体类别)
        :return:
        """

        weights_len, boxes_len = len(self.weights), len(raw_boxes)
        if weights_len != boxes_len:
            print(f'Warning: number of weights {weights_len} not equal to number of boxes {boxes_len},'
                  f'reset weights to all 1s.')
            self.weights = torch.ones(len(raw_boxes), dtype=self.weights.dtype)

        # 过滤掉置信度太低以及坐标不合法的bboxes，返回各个类别对应的bboxes
        # dict{class->boxes}
        # boxes are (n,8): each is (label,conf,weight,model_idx,x1,y1,x2,y2)
        filtered_boxes = self.filter_boxes(raw_boxes, scores, labels)
        if not len(filtered_boxes):
            # locations, scores, labels
            return torch.zeros((0, 4), dtype=torch.int, device=raw_boxes.device), \
                   torch.zeros(0, dtype=torch.float, device=scores.device), \
                   torch.zeros(0, dtype=torch.int, device=labels.device)

        fused_boxes = []
        # 依次处理各个类别
        for cls_boxes in filtered_boxes.values():
            # 每个簇代表各物体对应的多个bboxes
            clusters = []
            # 各个簇加权融合计算出来的bbox代表
            weighted_boxes = []

            # 找到每个bbox最匹配的簇，同时更新该簇中心(加权融合计算)
            # 加权融合后，得到的bbox的置信度是簇中所有bboxes的置信度均值；
            # 对应的模型权重是簇中所有bboxes的模型权重之和；
            # 坐标是簇中各bboxes的坐标的加权平均，权重是各bboxes置信度除以簇中所有bboxes的置信度之和
            for box in cls_boxes:
                idx, iou = self.box_matching(weighted_boxes, box)
                idx = int(idx)
                if idx != -1:
                    clusters[idx].append(box.detach())
                    weighted_boxes[idx] = self.calculate_weighted_box(clusters[idx])
                else:
                    clusters.append([box.detach()])
                    weighted_boxes.append(box.detach())
            assert len(clusters) == len(weighted_boxes)
            print(f'clusters:\n{clusters}\n')
            print(f'weighted_boxes:\n{weighted_boxes}\n')

            # rescale confidence
            for boxes, weighted_box in zip(clusters, weighted_boxes):
                if self.conf_type == 'box_and_model_avg':
                    weighted_box[1] *= len(boxes) / weighted_box[2]
                    # 由哪几个模型预测而来
                    model_idx = torch.unique(boxes[:, 3])
                    weighted_box[1] *= self.weights[model_idx].sum() / self.weights.sum()
                elif self.conf_type == 'absent_model_aware_avg':
                    # 由哪几个模型预测而来
                    model_idx = torch.unique(boxes[:, 3])
                    mask = torch.ones_like(self.weights, dtype=torch.bool)
                    # 通过掩码过滤来知道几个模型没有预测这个簇
                    mask[model_idx] = False
                    weighted_box[1] *= len(boxes) / (weighted_box[2] + self.weights[mask].sum())
                elif self.conf_type == 'avg' and not self.allows_overflow:
                    weighted_box[1] *= min(len(boxes), int(self.weights.sum())) / self.weights.sum()
                else:
                    # 当有模型在这个簇中预测多个bboxes时，这种情况计算得到的置信度可能大于1
                    weighted_box[1] *= len(boxes) / self.weights.sum()

            fused_boxes.append(torch.stack(weighted_boxes))

            del weighted_boxes
            del clusters

        # 将bboxes按置信度降序排序
        fused_boxes = torch.cat(fused_boxes)
        sorted_indices = fused_boxes[:, 1].argsort(descending=True)
        fused_boxes = fused_boxes[sorted_indices]

        # locations, scores, labels
        return fused_boxes[:, 4:].int(), fused_boxes[:, 1], fused_boxes[:, 0]


if __name__ == '__main__':
    box_fusion_module = BoxFusion()
    boxes = torch.tensor([
        [10, 15, 58, 70],
        [15, 25, 60, 82],
        [200, 75, 258, 224],
        [190, 95, 268, 250],
        [88, 200, 158, 400],
    ], dtype=torch.int)

    fig = plt.figure()
    ax = plt.subplot(121)
    ax.set_title('raw boxes')
    for box in boxes.numpy():
        w, h = box[2] - box[0], box[3] - box[1]
        w /= 512
        h /= 512
        x, y = box[0] / 512, box[1] / 512
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor='b')
        ax.add_patch(rect)

    boxes = boxes.unsqueeze(dim=0)
    scores = torch.tensor([[0.6, 0.4, 0.9, 0.8, 0.7]], dtype=torch.float)
    labels = torch.tensor([[0, 0, 1, 1, 2]], dtype=torch.int)
    print(f'boxes:\n{boxes}\n')
    print(f'scores:\n{scores}\n')
    print(f'labels:\n{labels}\n')

    fused_boxes, fused_scores, fused_labels = box_fusion_module.get_fusion_boxes(boxes, scores, labels)
    print(f'fused_boxes:\n{fused_boxes}\n')
    print(f'fused_scores:\n{fused_scores}\n')
    print(f'fused_labels:\n{fused_labels}\n')

    ax = plt.subplot(122)
    ax.set_title('fused boxes')
    for box in fused_boxes.numpy():
        w, h = box[2] - box[0], box[3] - box[1]
        w /= 512
        h /= 512
        x, y = box[0] / 512, box[1] / 512
        rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1, edgecolor='r')
        ax.add_patch(rect)

    plt.suptitle('weighted boxes fusion')
    plt.show()
