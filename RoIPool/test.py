import torch

from RoIPool.roi_align import RoIAlign
from RoIPool.roi_pooling import RoIPooling2D


if __name__ == '__main__':
    ratio = 32
    feat_size = 14
    img_size = feat_size * ratio

    feats = torch.randn(2, 3, feat_size, feat_size)
    N = feats.shape[0]
    # (batch_index,x1,y1,x2,y2)
    rois = torch.randint(0, img_size, (3, 5)).float()
    rois[:, 3:] = rois[:, 1:3] + 100
    rois[:, 1:].clamp_(min=0, max=img_size)
    assert (rois[:, 3:] > rois[:, 1:3]).all()
    num_rois = rois.shape[0]
    rois[:, 0] = torch.randint(0, N, (num_rois,)).float()
    print(f"RoIs:\n{rois}\n")

    roi_pool2d = RoIPooling2D(pooled_height=2, pooled_width=2)
    roi_align = RoIAlign(pooled_h=2, pooled_w=2)

    pooled_feats = roi_pool2d(feats, rois)
    print(f"pooled features(w RoI Pooling):\n{pooled_feats}\n")

    pooled_feats = roi_align(feats, rois)
    print(f"pooled features(w RoI Align):\n{pooled_feats}")
