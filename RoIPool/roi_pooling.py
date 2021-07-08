# RoI Pooling Implementation

import torch
import torch.nn as nn
# import torch.nn.functional as F


class RoIPooling2D(nn.Module):
    """RoI Pooling 2D Module"""
    def __init__(self, spacial_scale: float = 1. / 32, pooled_height: int = 7, pooled_width: int = 7):
        """
            :param spacial_scale: scale ratio -- size_feature / size_roi
            :param pooled_height: target height after pooling
            :param pooled_width: target width after pooling
        """
        super(RoIPooling2D, self).__init__()
        assert 0. < spacial_scale < 1., f"scale ratio should in range (0, 1), got{spacial_scale}"

        self.ratio = spacial_scale
        self.h = pooled_height
        self.w = pooled_width

    def forward(self, features, rois):
        """
        RoIModule 2D.
            :param features: (N, C, H, W) -> a batch of feature maps
            :param rois: (num_roi, 5) -> each roi is (batch_index, x_min, y_min, x_max, y_max)
            :return: pooled target with shape (num_roi, C, pooled_height, pooled_width)
        """
        _, C, H, W = features.shape
        num_rois = rois.shape[0]
        pooled_features = torch.zeros(num_rois, C, self.h, self.w,
                                      dtype=features.dtype, device=features.device)

        invalid_rois = []
        for idx, roi in enumerate(rois):
            # 1st quantization
            box = torch.floor(roi[1:] * self.ratio)
            # make its size in the range of feature map's size
            box[0::2].clamp_(min=0, max=W)
            box[1::2].clamp_(min=0, max=H)
            if not (box[2:] - box[:2] > 0).all():
                print("invalid bbox!")
                invalid_rois.append(roi)
                continue

            x, y = box[:2]
            bin_h, bin_w = (box[3] - box[1]) / self.h, (box[2] - box[0]) / self.w

            # Or u can use max_pooling2d function to replace your self-implementation code
            # batch_idx = roi[0]
            # feat = features[batch_idx]
            # x1, y1, x2, y2 = box.int()
            # pooled_features[idx] = F.max_pool2d(feat[:, y1:y2, x1:x2], (self.h, self.w))

            # Note that there are overlaps between adjacent bins
            for i in range(self.h):
                # 2nd quantization
                y1 = (y + torch.floor(i * bin_h)).int()
                y2 = (y + torch.ceil((i + 1) * bin_h)).int()

                y1.clamp_(min=0, max=H)
                y2.clamp_(min=0, max=H)
                if not y2 > y1:
                    print("invalid bin!")
                    continue

                for j in range(self.w):
                    # 2nd quantization
                    x1 = (x + torch.floor(j * bin_w)).int()
                    x2 = (x + torch.ceil((j + 1) * bin_w)).int()

                    x1.clamp_(min=0, max=W)
                    x2.clamp_(min=0, max=W)
                    if not x2 > x1:
                        print("invalid bin")
                        continue

                    batch_idx = roi[0].long()
                    feat = features[batch_idx]
                    # max pooling
                    pooled_features[idx, :, i, j] = feat[:, y1:y2, x1:x2].max(dim=1)[0].max(dim=1)[0]

        if invalid_rois:
            print(f"There are {len(invalid_rois)} invalid RoIs:\n"
                  f"{invalid_rois}")

        return pooled_features
