"""IoU Loss for FCOS evaluation, also contains Linear IoU Loss, GIoU Loss."""

import torch


class IoULoss:
    def __init__(self, loss_type='iou', reduction='mean'):
        self.loss_type = loss_type
        self.reduction = reduction

    def __call__(self, ltrb_preds, ltrb_targets, weight=None):
        # (n,2)
        lt_min = torch.min(ltrb_preds[:, :2], ltrb_targets[:, :2])
        rb_min = torch.min(ltrb_preds[:, 2:], ltrb_targets[:, 2:])
        wh_inter = lt_min + rb_min
        # (n,)
        inter = wh_inter[:, 0] * wh_inter[:, 1]

        # (n,2)
        wh_preds = ltrb_preds[:, :2] + ltrb_preds[:, 2:]
        # (n,)
        area_preds = wh_preds[:, 0] * wh_preds[:, 1]

        # (n,2)
        wh_targets = ltrb_targets[:, :2] + ltrb_targets[:, 2:]
        # (n,)
        area_targets = wh_targets[:, 0] * wh_targets[:, 1]

        # (n,)
        iou = inter / (area_preds + area_targets - inter).clamp(min=1e-6)
        if self.loss_type == 'iou':
            # (n,)
            loss = -iou.clamp(min=1e-6).log()
        elif self.loss_type == 'linear_iou':
            # (n,)
            loss = 1. - iou
        elif self.loss_type == 'giou':
            lt_max = torch.max(ltrb_preds[:, :2], ltrb_targets[:, :2])
            rb_max = torch.max(ltrb_preds[:, 2:], ltrb_targets[:, 2:])
            # (n,2)
            wh_union = lt_max + rb_max
            # (n,)
            union = wh_union[:, 0] * wh_union[:, 1]

            # (n,)
            giou = iou - (union - area_preds - area_targets) / union.clamp(min=1e-6)
            loss = 1. - giou
        else:
            raise NotImplementedError(f"not implemented iou type: '{self.loss_type}'")

        if weight is not None:
            loss *= weight

        if self.reduction == 'mean':
            # ()
            return loss.mean()
        elif self.reduction == 'sum':
            # ()
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise NotImplementedError(f"not implemented reduction: '{self.reduction}'")
