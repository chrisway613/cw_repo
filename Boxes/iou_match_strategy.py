# Label Assignment based on IoU strategy

import torch


def compute_iou(box_a, box_b):
    n_a = box_a.size(0)
    n_b = box_b.size(0)

    # (n_a, n_b, 2)
    max_xy = torch.min(
        box_a[:, None, 2:].expand(n_a, n_b, 2),
        box_b[None, :, 2:].expand(n_a, n_b, 2)
    )
    # (n_a, n_b, 2)
    min_xy = torch.max(
        box_a[:, None, :2].expand(n_a, n_b, 2),
        box_b[None, :, :2].expand(n_a, n_b, 2)
    )
    wh = (max_xy - min_xy).clamp(min=0)
    # (n_a, n_b)
    inter = wh[:, :, 0] * wh[:, :, 1]

    # (n_a,2)
    wh_a = box_a[:, 2:] - box_a[:, :2]
    # (n_a,)
    area_a = wh_a[:, 0] * wh_a[:, 1]

    # (n_b,2)
    wh_b = box_b[:, 2:] - box_b[:, :2]
    # (m_b,)
    area_b = wh_b[:, 0] * wh_b[:, 1]

    union = area_a[:, None] + area_b - inter
    return inter / union


if __name__ == '__main__':
    # 区分正负样本的IoU阀值
    threshold = .5
    # 3个目标物体，2个类别（不包括背景），10个先验框 (x1,y1,x2,y2)
    labels = torch.randint(0, 2, (3,))
    print(f'labels:\n{labels}\n')
    gt_boxes = torch.randn(3, 4).clamp_(min=0.)
    gt_boxes[:, 2:] = gt_boxes[:, :2] + torch.randint(1, 5, (gt_boxes.size(0), 2))
    print(f'gt boxes:\n{gt_boxes}')
    prior_boxes = torch.randn(10, 4).clamp_(min=0.)
    prior_boxes[:, 2:] = prior_boxes[:, :2] + torch.randint(1, 5, (prior_boxes.size(0), 2))
    print(f'prior boxes:\n{prior_boxes}')

    # (n_objects=3,n_priors=10) GT和先验bbox两两计算IoU
    overlaps = compute_iou(gt_boxes, prior_boxes)
    print(f"IoU:\n{overlaps}\n")
    # (n_objects,) best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(dim=1)
    print(f'best prior indices:\n{best_prior_idx}\n')
    # (n_priors,) best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)
    print(f'best truth IoU:\n{best_truth_overlap}\n')

    # 将与每个gt有最大IoU的先验框（下称为最优先验框）与其匹配的gt的IoU设置为一个无限制（由于IoU最大只能为1，因此这里设置为2就足够了）
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    print(f'best truth IoU:\n{best_truth_overlap}\n')
    # ensure every gt matches with its prior of max overlap
    # 同时将每个最优先验框的gt，形成对称形式
    # （因为仅仅基于IoU计算可能出现这种情况：与gt1有最大IoU的是prior1，但prior1与所有gt计算IoU时可能与gt2有最大IoU）
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    print(f'best truth indices:\n{best_truth_idx}\n')

    # 选出每个先验框匹配到的gt
    matches = gt_boxes[best_truth_idx]  # Shape: [num_priors,4]
    # +1是由于原标签中不包含背景类，因此留出0给背景用
    conf = labels[best_truth_idx] + 1  # Shape: [num_priors]
    # 将最大IoU小于阀值的样本标签置为背景类
    conf[best_truth_overlap < threshold] = 0  # label as background
    print(f'matched gt boxes:\n{matches}\n\nmatched labels:\n{conf}')
