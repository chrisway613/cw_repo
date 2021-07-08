# RoI Align Implementation

import math

import torch
import torch.nn as nn
# import torch.nn.functional as F


class RoIAlign(nn.Module):
    """RoIAlign Module"""

    @classmethod
    def sampling_feats(cls, feats, sampled_pos):
        C, H, W = feats.shape
        xs, ys = sampled_pos.shape[:2]
        sampled_feats = torch.zeros(C, ys, xs, dtype=feats.dtype, device=feats.device)

        for i in range(xs):
            for j in range(ys):
                x, y = sampled_pos[i, j]
                x1 = torch.floor(x).int().clamp_(min=0, max=W - 1)
                x2 = (x1 + 1).clamp_(min=0, max=W - 1)
                y1 = torch.floor(y).int().clamp_(min=0, max=H - 1)
                y2 = (y1 + 1).clamp_(min=0, max=H - 1)

                mid_top = (x - x1) * feats[:, y1, x2] + (x2 - x) * feats[:, y1, x1]
                mid_bottom = (x - x1) * feats[:, y2, x2] + (x2 - x) * feats[:, y2, x1]
                sampled_xy_feat = (y - y1) * mid_bottom + (y2 - y) * mid_top
                sampled_feats[:, j, i] = sampled_xy_feat

        return sampled_feats

    def __init__(self, spacial_ratio: float = 1. / 32, pooled_h: int = 7, pooled_w: int = 7, n_sampled: int = 4):
        """
        :param spacial_ratio: scale ratio -- size_feature / size_roi
        :param pooled_h: target height after pooling
        :param pooled_w: target width after pooling
        """
        super(RoIAlign, self).__init__()
        assert 0. < spacial_ratio < 1., f"scale ratio should in range (0, 1), got{spacial_ratio}"
        assert n_sampled > 0, f"sample counts should more than 0, got {n_sampled}"

        self.ratio = spacial_ratio
        self.h = pooled_h
        self.w = pooled_w
        self.sub_bin_size = int(math.sqrt(n_sampled))

    def forward(self, features, rois):
        _, C, H, W = features.shape
        num_rois = rois.shape[0]
        pooled_features = torch.zeros(num_rois, C, self.h, self.w, dtype=features.dtype, device=features.device)

        invalid_rois = []
        for idx, roi in enumerate(rois):
            # no quantization
            box = roi[1:] * self.ratio
            box[0::2].clamp_(min=0, max=W)
            box[1::2].clamp_(min=0, max=H)
            if not (box[2:] - box[:2] > 0).all():
                print("invalid bounding box!")
                invalid_rois.append(roi)
                continue

            x, y = box[:2]
            box_w, box_h = box[2:] - box[:2]
            bin_w, bin_h = box_w / self.w, box_h / self.h

            for i in range(self.h):
                # no quantization
                y1 = y + i * bin_h
                y2 = y1 + bin_h

                y1.clamp_(min=0, max=H)
                y2.clamp_(min=0, max=H)
                if not y2 > y1:
                    print("invalid bin!")
                    continue

                for j in range(self.w):
                    # no quantization
                    x1 = x + j * bin_w
                    x2 = x1 + bin_w

                    x1.clamp_(min=0, max=W)
                    x2.clamp_(min=0, max=W)
                    if not x2 > x1:
                        print("invalid bin!")
                        continue

                    # sub-bin division
                    sub_x, sub_y = x1, y1
                    sub_bin_w, sub_bin_h = bin_w / self.sub_bin_size, bin_h / self.sub_bin_size
                    # center point of the top-left sub-bin
                    sub_cx, sub_cy = sub_x + sub_bin_w / 2, sub_y + sub_bin_h / 2
                    # sampled points of each sub-bin
                    sampled_points = torch.zeros([self.sub_bin_size] * 2 + [2], device=box.device)
                    for m in range(self.sub_bin_size):
                        sub_m_cx = sub_cx + m * sub_bin_w
                        for n in range(self.sub_bin_size):
                            sub_n_cy = sub_cy + n * sub_bin_h
                            sampled_points[m, n] = torch.as_tensor([sub_m_cx, sub_n_cy], device=box.device)

                    batch_idx = roi[0].long()
                    # (C,sub_bin_size,sub_bin_size) interpolated sampled point features
                    sampled_feat = self.sampling_feats(features[batch_idx], sampled_points)
                    pooled_features[idx, :, i, j] = sampled_feat.max(dim=1)[0].max(dim=1)[0]
                    # Or u can use max_pooling2d function
                    # pooled_features[idx, :, i, j] = F.adaptive_avg_pool2d(sampled_feat, 1).squeeze()

        if invalid_rois:
            print(f"There are {len(invalid_rois)} invalid RoIs:\n"
                  f"{invalid_rois}")

        return pooled_features
