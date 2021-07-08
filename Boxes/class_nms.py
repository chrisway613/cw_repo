"""为各类别检测框坐标加上对应的位移，使得不同类别的框不会相互重叠，从而在一个方法中达到对各类别依次进行NMS的效果。"""

import torch


def nms(boxes, scores, threshold=.5):
    if not len(boxes):
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    keep = []
    _, indices = scores.sort(descending=True)
    while indices.numel() > 0:
        if indices.numel() == 1:
            keep.append(indices.item())
            break

        i = indices[0].item()
        keep.append(i)
        indices = indices[1:]

        # clamp的值要求是number不能是tensor，因此要调用.item()
        xmin = x1[indices].clamp(min=x1[i].item())
        ymin = y1[indices].clamp(min=y1[i].item())
        xmax = x2[indices].clamp(max=x2[i].item())
        ymax = y2[indices].clamp(max=y2[i].item())

        inter = ((xmax - xmin + 1) * (ymax - ymin + 1)).clamp(min=0)
        union = (areas[indices] + areas[i] - inter).clamp(min=1e-6)
        iou = inter / union

        indices = indices[iou <= threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def cls_nms(boxes, scores, cls_indices, threshold=.5):
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    offsets = cls_indices * (1 + boxes.max())
    boxes_nms = boxes + offsets[:, None]

    return nms(boxes_nms, scores, threshold=threshold)
