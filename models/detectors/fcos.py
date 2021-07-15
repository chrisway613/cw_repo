"""FCOS Implementation: Model, Loss, Post Process"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.resnet import resnet50
from models.backbones.darknet import DarkNet19

from Boxes.class_nms import cls_nms
from Train.loss.iou_loss import IoULoss


class FPN(nn.Module):
    """Feature Pyramid Network used for FCOS, assuming features come from DarkNet19 or ResNet50"""

    def __init__(self, planes=256, use_p5=True, backbone='resnet50'):
        super(FPN, self).__init__()

        # backbone输出的3层特征通道
        if backbone == 'resnet50':
            in_planes = [512, 1024, 2048]
        elif backbone == 'darknet19':
            in_planes = [256, 512, 1024]
        else:
            raise NotImplementedError(f"backbone only support 'resnet50' or 'darknet19', current: '{backbone}'")

        # 横向连接
        self.prj5 = nn.Conv2d(in_planes[2], planes, 1)
        self.prj4 = nn.Conv2d(in_planes[1], planes, 1)
        self.prj3 = nn.Conv2d(in_planes[0], planes, 1)

        # top-down
        self.conv5 = nn.Conv2d(planes, planes, 3, padding=1)
        self.conv4 = nn.Conv2d(planes, planes, 3, padding=1)
        self.conv3 = nn.Conv2d(planes, planes, 3, padding=1)
        if use_p5:
            self.conv6 = nn.Conv2d(planes, planes, 3, stride=2, padding=1)
        else:
            raise ValueError(f"p6 and p7 can only be built on top of p5")
        self.conv7 = nn.Conv2d(planes, planes, 3, stride=2, padding=1)

        # self.apply()会对self.children()返回的模块调用传进去的参数指定的方法(这里是self._init_conv)
        self.apply(self._init_conv)

    @staticmethod
    def _init_conv(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, features):
        c3, c4, c5 = features

        # 1. 将backbone输出的特征图通道数映射到一致
        p5 = self.prj5(c5)
        p4 = self.prj4(c4)
        p3 = self.prj3(c3)

        # 2. 特征融合
        p4 += F.interpolate(p5, size=p4.shape[-2:])
        p3 += F.interpolate(p4, size=p3.shape[-2:])

        # 3. 卷积平滑(融合后的)特征
        p3 = self.conv3(p3)
        p4 = self.conv4(p4)
        p5 = self.conv5(p5)

        # 4. 由P5进一步下采样得到P6、P7
        p6 = self.conv6(p5)
        # 注意P6先经过ReLU激活下采样得到P7
        p7 = self.conv7(F.relu(p6))

        return p3, p4, p5, p6, p7


class Scales(nn.Module):
    """Scaling regression outputs of FCOS head
       为FCOS各特征层设置一个可学习的标量，去自适应回归的尺度"""

    def __init__(self, value=1.):
        super(Scales, self).__init__()
        self.scale = nn.Parameter(torch.tensor([value]))

    def forward(self, x):
        return self.scale * x


class FCOSHead(nn.Module):
    """Detection Head of FCOS"""

    def __init__(self, num_classes, channels=256, gn=True, cnt_on_reg=True, norm_reg_targets=False,
                 prior=0.01, num_convs=4):
        """

        :param num_classes: 目标类别数；
        :param channels: 特征通道数；
        :param gn: 是否使用Group Normalization；
        :param cnt_on_reg: Center-ness分支是否搭在回归分支上；
        :param norm_reg_targets: 是否对回归目标(使用下采样步长)归一化；
        :param prior: 先验概率；
        :param num_convs: 分类和回归分支的卷积层数量
        """

        super(FCOSHead, self).__init__()

        self.cnt_on_reg = cnt_on_reg
        self.norm_reg_targets = norm_reg_targets

        # 分类、回归分支，结构一致
        cls_branch = []
        reg_branch = []
        for _ in range(num_convs):
            cls_branch.append(nn.Conv2d(channels, channels, 3, padding=1))
            reg_branch.append(nn.Conv2d(channels, channels, 3, padding=1))
            if gn:
                cls_branch.append(nn.GroupNorm(32, channels))
                reg_branch.append(nn.GroupNorm(32, channels))
            cls_branch.append(nn.ReLU(inplace=True))
            reg_branch.append(nn.ReLU(inplace=True))
        self.add_module('cls_branch', nn.Sequential(*cls_branch))
        self.add_module('reg_branch', nn.Sequential(*reg_branch))

        # 预测头部，注意是3x3卷积而非1x1
        self.cls_logits = nn.Conv2d(channels, num_classes, 3, padding=1)
        self.bbox_pred = nn.Conv2d(channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(channels, 1, 3, padding=1)

        self.apply(self._init_conv)
        # 将分类头的卷积偏置设置为先验概率
        self._init_cls_logits(prob=prior)

        # 为各特征层设置一个可学习的标量，去自适应回归的尺度
        self.scales = nn.ModuleList([Scales() for _ in range(5)])

    @staticmethod
    def _init_conv(m):
        if isinstance(m, nn.Conv2d):
            # mean=0
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                # fill with 0
                nn.init.zeros_(m.bias)

    def _init_cls_logits(self, prob=0.01):
        # make bias be the prior probability
        nn.init.constant_(self.cls_logits.bias, -math.log((1. - prob) / prob))

    def forward(self, inputs):
        # 5层特征
        assert len(inputs) == len(self.scales)

        # 分别记录各层特征点的预测结果
        logits = []
        bbox_reg = []
        centerness = []

        for p, s in zip(inputs, self.scales):
            cls_branch_out = self.cls_branch(p)
            cls_logits = self.cls_logits(cls_branch_out)
            logits.append(cls_logits)

            reg_branch_out = self.reg_branch(p)
            # 注意要将网络的回归输出映射成非负值
            # 可使用exp()或relu
            # 若使用relu的话则同时要对回归的目标进行归一化
            bbox_pred = s(self.bbox_pred(reg_branch_out))
            bbox_reg.append(F.relu(bbox_pred) if self.norm_reg_targets else torch.exp(bbox_pred))

            centerness.append(self.centerness(reg_branch_out) if self.cnt_on_reg else self.centerness(cls_branch_out))

        return logits, bbox_reg, centerness


class DefaultConfig:
    """Default configuration of FCOS"""

    # 一个极大值
    INF = 1e8

    # backbone
    backbone = "darknet19"
    pretrained = False
    weight_file = None

    # fpn
    use_p5 = True
    fpn_out_channels = 256

    # head
    prior = 0.01
    class_num = 20
    cnt_on_reg = True
    use_GN_head = True
    norm_reg_targets = False

    # training
    # FPN 5层特征对应的下采样步长
    strides = [8, 16, 32, 64, 128]
    # 每层特征负责预测的物体尺度(bbox 边长)范围
    obj_sizes_of_interest = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, INF]]

    # loss
    focal_loss_alpha = .25
    focal_loss_gamma = 2.
    # 是否开启中心采样策略
    cnt_sampling = False
    cnt_sampling_radius = 1.5
    # 还可以是iou和linear_iou
    iou_loss_type = 'giou'

    # inference
    # 进行nms先剔除掉分类得分过低的结果
    pre_nms_score_threshold = 0.05
    # nms前每张图片最多不超过这么多个预测结果
    pre_nms_top_n = 1000
    # nms时两个预测框的IoU超过这个值则认为是重叠框
    nms_iou_threshold = 0.6
    # 每张图片最多检测的物体数量
    max_detection_boxes_num = 100
    # 所有检测框的边长必须大于这个值
    box_min_size = 0


class FCOS(nn.Module):
    def __init__(self, cfg=None):
        super(FCOS, self).__init__()

        if cfg is None:
            cfg = DefaultConfig()
        self.cfg = cfg

        if cfg.backbone == 'resnet50':
            self.backbone = resnet50(pretrained=cfg.pretrained, include_top=False)
        elif cfg.backbone == 'darknet19':
            self.backbone = DarkNet19(include_top=False, pretrained=cfg.pretrained, weight_file=cfg.weight_file)
        else:
            raise NotImplementedError(f"backbone only support 'resnet50' of 'darknet19', current: '{cfg.backbone}'")

        self.fpn = FPN(planes=cfg.fpn_out_channels, use_p5=cfg.use_p5, backbone=cfg.backbone)
        self.head = FCOSHead(cfg.class_num, gn=cfg.use_GN_head,
                             cnt_on_reg=cfg.cnt_on_reg, norm_reg_targets=cfg.norm_reg_targets, prior=cfg.prior)

    def forward(self, x):
        cs = self.backbone(x)
        # backbone输出3层特征
        assert len(cs) == 3
        feats = self.fpn(cs)
        # FPN输出5层特征
        assert len(feats) == 5
        cls_logits, box_reg, centerness = self.head(feats)

        # 5个特征层的预测结果
        # 注意，cls_logits和centerness都是logits
        # 后续需要用Sigmoid或Softmax转换
        return cls_logits, box_reg, centerness


def fmap2img_locations(feature_levels, strides):
    """计算特征图上的每点对应到输入图像的位置，位置形式用(x,y)表示"""

    assert len(feature_levels) == len(strides)

    # len=n_levels
    loc_levels = []
    for feat, s in zip(feature_levels, strides):
        h, w = feat.shape[-2:]
        # (h,)
        y = torch.arange(0, h * s, s, dtype=torch.float, device=feat.device)
        # (w,)
        x = torch.arange(0, w * s, s, dtype=torch.float, device=feat.device)

        # (h,w) (h,w)
        ys, xs = torch.meshgrid(y, x)
        # (h*w,)
        ys = ys.flatten()
        # (h*w,)
        xs = xs.flatten()

        # (h*w,2)
        loc = torch.stack((xs, ys), dim=-1) + s // 2
        loc_levels.append(loc)

    return loc_levels


class FCOSPostProcessor:
    """
    FCOS的后处理过程，包括：
       i).   剔除分类得分过低的；
       ii).  将分类得分与centerness相乘并且每张图选择分数最高的前1000个预测结果；
       iii). 移除尺寸过小的检测框；
       iv).  将检测框坐标限制到输入图像坐标空间内；
       v).   NMS；
       vi).  每张图选择分数最高的前100个检测结果
    """

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DefaultConfig()

        self.strides = cfg.strides
        self.norm_reg_targets = cfg.norm_reg_targets

        self.pre_nms_score_threshold = cfg.pre_nms_score_threshold
        self.pre_nms_top_n = cfg.pre_nms_top_n

        self.nms_threshold = cfg.nms_iou_threshold
        self.max_det_num = cfg.max_detection_boxes_num

        self.box_min_size = cfg.box_min_size

    def __call__(self, cls_logits, box_reg, centerness, img_sizes):
        # 预测结果是各个特征层的
        assert len(cls_logits) == len(box_reg) == len(centerness) == len(self.strides)

        # 1. 计算特征点位于输入图像上的位置

        # list 其中每项是shape为(h_level*w_level,2)的张量，代表各层特征点在输入图像上的位置 (x,y)
        loc_levels = fmap2img_locations(cls_logits, self.strides)
        # (n_points,2) 每一个为位置是(x,y)形式
        locs = torch.cat(loc_levels, dim=0)

        # 2. 维度变换、获取正样本候选索引、将分类得分与centerness相乘的结果作为得分

        b, num_cls = cls_logits[0].shape[:2]

        cls_logits = [logits_per_level.permute(0, 2, 3, 1).reshape(b, -1, num_cls)
                      for logits_per_level in cls_logits]
        # (b,n_points,num_cls)
        cls_score_batched = torch.cat(cls_logits, dim=1).sigmoid()
        # 测试时可以打开以下注释
        # cls_score_batched[0, :(locs.size(0) // 2), 1] = .75
        # cls_score_batched[1, (locs.size(0) // 2):, -5:] = .66
        # (b,n_points,num_cls) bool
        pos_batched = cls_score_batched > self.pre_nms_score_threshold

        box_reg = [reg_per_level.permute(0, 2, 3, 1).reshape(b, -1, 4) for reg_per_level in box_reg]
        # 注意，若训练时使用了下采样步长归一化回归标签，那么推理时需要将回归量乘上步长以进行解码
        if self.norm_reg_targets:
            assert len(box_reg) == len(self.strides)
            box_reg = [reg_per_level * s for reg_per_level, s in zip(box_reg, self.strides)]
        # (b,n_points,4)
        reg_batched = torch.cat(box_reg, dim=1)

        centerness = [cnt_per_level.permute(0, 2, 3, 1).reshape(b, -1, 1) for cnt_per_level in centerness]
        # (b,n_points,1)
        cnt_score_batched = torch.cat(centerness, dim=1).sigmoid()

        # (b,n_points,num_cls) 将分类得分与centerness相乘
        score_batched = cls_score_batched * cnt_score_batched

        # 每张图的预测结果
        # 其中每项是(n_det,6), 6 -- x1,y1,x2,y2,cls,score
        predictions = []
        for score_img, reg_img, pos_img, size_img in zip(score_batched, reg_batched, pos_batched, img_sizes):
            # 该图中所有预测结果的分类得分都低于0.05，代表没有检测到任何物体
            if pos_img.sum() == 0:
                # print("no objects detected")
                predictions.append(torch.zeros((0, 6), dtype=torch.float, device=reg_img.device))
                continue

            # 3. 剔除掉分类得分小于0.05的预测结果

            # (pos_img.sum(),) 正样本得分
            score_pos = score_img[pos_img]
            # 正样本位置点索引 正样本类别
            # 注意，同个位置可能出现多次，因为在多个类别上可能重复出现
            # (pos_img.sum(),)， (pos_img.sum(),)
            pos_points_ind, cls_pos = pos_img.nonzero(as_tuple=True)
            # 0留给背景
            cls_pos += 1
            # (pos_img.sum(),4)
            reg_pos = reg_img[pos_points_ind]
            # (pos_img.sum(),2)
            locs_pos = locs[pos_points_ind]

            # 4. 选择得分最高的前1000个预测结果

            # 记n_pos = min(pos_img.sum(), pre_nms_top_n)
            if pos_img.sum() > self.pre_nms_top_n:
                # (n_pos,), (n_pos,)
                score_pos, topk_ind = score_pos.topk(self.pre_nms_top_n)
                # (n_pos,)
                cls_pos = cls_pos[topk_ind]
                # (n_pos,4)
                reg_pos = reg_pos[topk_ind]
                # (n_pos,2)
                locs_pos = locs_pos[topk_ind]

            # 5. 移除尺寸过小的框

            # (n_pos,2)
            x1y1 = locs_pos - reg_pos[:, :2]
            # (n_pos,2)
            x2y2 = locs_pos + reg_pos[:, 2:]
            # (n_pos,2)
            wh = x2y2 - x1y1 + 1
            keep = wh.min(dim=-1)[0] > self.box_min_size

            # (n_kept,) 注意这里要开方 还原数值尺度到[0,1] 因为之前分类得分与centerness乘在了一起
            scores = score_pos[keep].sqrt()
            # (n_kept,) 将labels转换成float是为了后续能和boxes、scores拼接起来
            labels = cls_pos[keep].float()
            # (n_kept,4)
            boxes = torch.cat((x1y1, x2y2), dim=-1)[keep]

            # 6. 将预测框坐标裁剪到输入图像的坐标范围内

            w_img, h_img = size_img
            boxes[:, [0, 2]].clamp_(min=0, max=w_img - 1)
            boxes[:, [1, 3]].clamp_(min=0, max=h_img - 1)

            # (n_kept,6) 将预测框、类别、得分拼接成一个张量作为一张图片的预测结果
            # 对labels和scores分别增加1个维度以便拼接起来
            pred_img = torch.cat((boxes, labels[:, None], scores[:, None]), dim=-1)

            # 7. NMS
            keep = cls_nms(boxes, scores, labels, threshold=self.nms_threshold)
            # (n_kept_post_nms,6)
            pred_img = pred_img[keep]

            # 8. 选择分数最高的top-100个
            if len(keep) > self.max_det_num:
                scores = pred_img[:, -1]
                # thresh是由小到大排第k的分数
                thresh, _ = torch.kthvalue(scores, len(keep) - self.max_det_num)
                keep = scores > thresh
                pred_img = pred_img[keep]

            # 将之前转换成float类型的类别索引(注意，从1开始)还原回来
            pred_img[:, -2] = pred_img[:, -2].long()
            predictions.append(pred_img)

        return predictions


class SigmoidFocalLoss:
    """FCOS的分类损失，使用Sigmoid的多分类Focal Loss"""

    def __init__(self, alpha=.25, gamma=2., reduction='mean'):
        assert reduction in ('mean', 'sum', 'none')

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def __call__(self, logits, targets):
        # 第一个维度等于特征点数量
        assert logits.size(0) == targets.size(0)
        assert logits.ndim == 2 and targets.ndim == 1

        # 标签值的最大值不能超过前景类别数，logits的输出通道数就是前景类别数量
        assert targets.max() <= logits.size(-1)

        num_cls = logits.size(-1)
        # (num_cls,) 这里需要从1开始，因为标签中的0是背景
        fg_cls_range = torch.arange(1, num_cls + 1, dtype=torch.float, device=logits.device)

        # (n,num_cls) bool
        pos = targets[:, None] == fg_cls_range[None, :]
        # (n,num_cls) bool
        neg = ~pos

        # (n,num_cls) 注意设置一个极小值下限，避免输入到log函数时溢出
        probs = logits.sigmoid().clamp(1e-6)
        pos_term = pos.float() * self.alpha * (1. - probs) ** self.gamma * probs.log()
        neg_term = neg.float() * (1. - self.alpha) * probs ** self.gamma * (1. - probs).log()

        # (n,num_cls)
        loss = -(pos_term + neg_term)
        if self.reduction == 'mean':
            # ()
            return loss.mean()
        elif self.reduction == 'sum':
            # ()
            return loss.sum()
        else:
            # (n,num_cls)
            return loss


class FCOSLossEvaluator:
    """FCOS的loss计算过程，包括标签分配"""

    INF = 1e8

    def __init__(self, cfg=None):
        if cfg is None:
            cfg = DefaultConfig()

        self.strides = cfg.strides
        self.obj_sizes_of_interest = cfg.obj_sizes_of_interest

        self.cnt_sampling = cfg.cnt_sampling
        self.cnt_sampling_radius = cfg.cnt_sampling_radius

        self.norm_reg_targets = cfg.norm_reg_targets

        # 注意以下损失函数中reduction都要是sum，因为后续要除以正样本数目(回归损失是用centerness标签值的总和)去算均值
        self.cls_loss = SigmoidFocalLoss(alpha=cfg.focal_loss_alpha, gamma=cfg.focal_loss_gamma, reduction='sum')
        self.reg_loss = IoULoss(loss_type=cfg.iou_loss_type, reduction='sum')
        self.cnt_loss = nn.BCEWithLogitsLoss(reduction='sum')

        self.INF = cfg.INF

    def __call__(self, outs, gts):
        cls_logits, box_reg, centerness = outs
        # 预测结果应该是5层特征层的
        assert len(cls_logits) == len(box_reg) == len(centerness)
        # (b,num_cls,h,w) (b,4,h,w) (b,1,h,w)
        assert cls_logits[0].ndim == box_reg[0].ndim == centerness[0].ndim == 4
        # 一个批次的图片数量
        assert cls_logits[0].size(0) == box_reg[0].size(0) == centerness[0].size(0) == len(gts)

        b = len(gts)
        num_cls = cls_logits[0].size(1)

        # 0. 计算特征点位于输入图像的位置

        # list len=n_levels 其中每项是(h_i*w_i,2) 每点的位置形式是(x,y)
        loc_levels = fmap2img_locations(cls_logits, self.strides)
        # 各层的特征点数量
        n_points_per_level = [loc_per_level.size(0) for loc_per_level in loc_levels]
        # 所有层的特征点数量
        n_points = sum(n_points_per_level)

        # 1. 标签分配

        # (b,n_points) (b,n_points,4)
        labels, reg_targets = self._label_assign(loc_levels, gts)
        assert labels.size(1) == reg_targets.size(1) == n_points
        # (b*n_points,)
        labels_flatten = labels.flatten()
        # (b*n_points,4)
        reg_targets_flatten = reg_targets.reshape(-1, 4)

        # 2. 维度变换

        cls_logits = [logits.permute(0, 2, 3, 1).reshape(b, -1, num_cls) for logits in cls_logits]
        # (b,n_points,num_cls)
        cls_logits_batched = torch.cat(cls_logits, dim=1)
        assert cls_logits_batched.size(1) == n_points
        # (b*n_points,num_cls)
        cls_logits_flatten = cls_logits_batched.reshape(-1, num_cls)

        box_reg = [reg.permute(0, 2, 3, 1).reshape(b, -1, 4) for reg in box_reg]
        # (b,n_points,4)
        box_reg_batched = torch.cat(box_reg, dim=1)
        assert box_reg_batched.size(1) == n_points
        # (b*n_points,4)
        box_reg_flatten = box_reg_batched.reshape(-1, 4)

        centerness = [cnt.permute(0, 2, 3, 1).reshape(b, -1, 1) for cnt in centerness]
        # (b,n_points,1)
        cnt_batched = torch.cat(centerness, dim=1)
        assert cnt_batched.size(1) == n_points
        # (b*n_points)
        cnt_flatten = cnt_batched.flatten()

        # 3. 计算正样本数量

        # 类别标签值为0代表背景
        pos = labels_flatten > 0
        num_pos = pos.sum().item()

        # 4. 计算分类损失

        # 注意对分母做个下限值的截断，避免在没有正样本时分母为0而导致溢出
        cls_loss = self.cls_loss(cls_logits_flatten, labels_flatten) / max(num_pos, 1.)

        if num_pos:
            # 5. 计算centerness损失

            # (n_pos,4)
            reg_targets = reg_targets_flatten[pos]
            # (n_pos,) 利用回归标签计算centerness标签
            cnt_targets = self.compute_cnt_targets(reg_targets)
            cnt_loss = self.cnt_loss(cnt_flatten[pos], cnt_targets) / num_pos

            # 6. 计算回归损失

            # 注意，回归损失是用centerness标签值加权和归一化
            reg_loss = self.reg_loss(box_reg_flatten[pos], reg_targets, weight=cnt_targets) / cnt_targets.sum()
        # 若没有正样本则回归损失和centerness损失均为0
        else:
            reg_loss = box_reg_flatten[pos].sum()
            cnt_loss = cnt_flatten[pos].sum()

        return cls_loss, reg_loss, cnt_loss

    @staticmethod
    def compute_cnt_targets(regression_targets):
        """根据各特征点的回归标签值来计算centerness的标签值"""

        lt = regression_targets[:, [0, 2]]
        rb = regression_targets[:, [1, 3]]
        cnt = torch.sqrt((lt.min(dim=-1)[0] * rb.min(dim=-1)[0]) / (lt.max(dim=-1)[0] * rb.max(dim=-1)[0]))

        return cnt

    def _label_assign(self, locs, gts):
        """FCOS的标签分配策略"""

        # 5层特征
        assert len(locs) == len(self.obj_sizes_of_interest)

        # 各层的特征点数量
        n_points_per_level = [loc_per_level.size(0) for loc_per_level in locs]
        # 计算所有层一共有多少个特征点
        n_points = sum(n_points_per_level)
        # (n_points,2) 将所有层的特征点位置拼接起来
        locs_all_levels = torch.cat(locs, dim=0)

        # 记录每张图片的标签
        cls_targets = []
        reg_targets = []
        for gt in gts:
            # (n_objs,4) (n_objs,)
            boxes, classes = gt[:, :-1], gt[:, -1]

            # 0. 计算每个特征点到每个物体框4条边的距离：ltrb

            # (n_points,n_objs,2)
            lt = locs_all_levels[:, None, :] - boxes[None, :, :2]
            # (n_points,n_objs,2)
            rb = boxes[None, :, 2:] - locs_all_levels[:, None, :]
            # (n_points,n_objs,4)
            ltrb = torch.cat((lt, rb), dim=-1)
            assert ltrb.size(0) == n_points

            # 1. 判断哪些位置点处于物体框内

            # (n_points,n_objs) bool
            in_boxes = ltrb.min(dim=-1)[0] > 0

            # 2. 根据各层回归的尺度，判断哪些物体是特征点应该负责预测的

            assert ltrb.size(0) == n_points
            ltrb_all_levels = ltrb.split(n_points_per_level)
            assert len(ltrb_all_levels) == len(self.obj_sizes_of_interest)

            in_levels = []
            for ltrb_per_level, obj_size_per_level in zip(ltrb_all_levels, self.obj_sizes_of_interest):
                min_size, max_size = obj_size_per_level
                ltrb_max = ltrb_per_level.max(dim=-1)[0]
                # (n_points_per_level,n_objs) bool
                in_level_mask = (ltrb_max >= min_size) & (ltrb_max <= max_size)
                in_levels.append(in_level_mask)
            # (n_points,n_objs) bool
            in_levels = torch.cat(in_levels, dim=0)

            # (n_points,n_objs) bool 判断每个特征点对每个物体是否为正样本
            pos = in_boxes & in_levels

            # 3. 中心采样

            if self.cnt_sampling:
                # 各层的采样半径
                radius_all_levels = [self.cnt_sampling_radius * s for s in self.strides]

                # (n_objs,2)
                obj_cnt = (boxes[:, :2] + boxes[:, 2:]) / 2
                # (n_points,n_objs)
                dis_obj_cnt = (locs_all_levels[:, None, :] - obj_cnt[None, :, :]).abs().max(dim=-1)[0]
                dis_obj_cnt_all_levels = dis_obj_cnt.split(n_points_per_level)
                assert len(dis_obj_cnt_all_levels) == len(radius_all_levels)

                in_cnts = []
                for dis_obj_cnt, radius in zip(dis_obj_cnt_all_levels, radius_all_levels):
                    # (n_points_per_level,n_objs) bool
                    cnt_mask = dis_obj_cnt < radius
                    in_cnts.append(cnt_mask)
                # (n_points,n_objs) bool
                in_cnts = torch.cat(in_cnts, dim=0)

                # (n_points,n_objs) bool 加入中心采样后，进一步判断每个特征点对每个物体是否为正样本
                pos &= in_cnts

            # 4. 每个特征点选取面积最小的物体作为目标

            # (n_points,n_objs,2)
            wh = ltrb[:, :, :2] + ltrb[:, :, 2:]
            # (n_points,n_objs)
            areas = wh[..., 0] * wh[..., 1]
            # 将负样本对应的目标物体面积置于无穷大
            areas[~pos] = self.INF
            # 每个特征点负责预测的目标物体及其面积 (n_points,) (n_points,)
            point_to_gt_area, point_to_gt_ind = areas.min(dim=-1)

            # (n_points,) 每个特征点的类别标签
            cls_targets_img = classes[point_to_gt_ind]
            # 将负样本的类别标签设置为背景类
            cls_targets_img[point_to_gt_area == self.INF] = 0
            cls_targets.append(cls_targets_img)

            # (n_points,4) 每个特征点的回归标签
            reg_targets_img = ltrb[range(n_points), point_to_gt_ind]
            # 将负样本的回归标签设置为无效值
            reg_targets_img[point_to_gt_area == self.INF] = -1
            reg_targets.append(reg_targets_img)
        # (b,n_points)
        cls_targets = torch.stack(cls_targets)
        # (b,n_points,4)
        reg_targets = torch.stack(reg_targets)

        # 5. 归一化回归标签

        if self.norm_reg_targets:
            reg_targets_all_levels = reg_targets.split(n_points_per_level, dim=1)
            assert len(reg_targets_all_levels) == len(self.strides)

            # 各层使用其下采样步长对回归标签进行归一化，映射到对应层的坐标尺度
            for reg_targets_per_level, s in zip(reg_targets_all_levels, self.strides):
                # (b,n_points_per_level,4)
                reg_targets_per_level /= s
            # (b,n_points,4)
            reg_targets = torch.cat(reg_targets_all_levels, dim=1)

        return cls_targets, reg_targets


if __name__ == '__main__':
    import random
    import numpy as np

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        # manual_seed()是仅对当前使用的GPU设置
        # 而manual_seed_all()是对所有GPU设置
        torch.cuda.manual_seed_all(0)
        # 程序起始就预先搜索各种算子(如卷积)对应的最优算法
        # 在网络结构和输入数据不变时，后续就不需要每次再进行搜索，从而加快训练速度
        torch.backends.cudnn.benchmark = True
        # 固定住cudnn的随机性
        torch.backends.cudnn.deterministic = True

    # backbone = DarkNet19(include_top=False)
    # fpn = FPN(backbone='darknet19')
    # # backbone = resnet50(include_top=False)
    # # fpn = FPN(backbone='resnet50')
    #
    imgs = torch.randn(2, 3, 320, 480)
    img_sizes = torch.stack([torch.tensor([img.size(-1), img.size(-2)]) for img in imgs])
    #
    # feats = backbone(x)
    # assert len(feats) == 3
    # for f in feats:
    #     print(f.shape)
    #
    # ps = fpn(feats)
    # for p in ps:
    #     print(p.shape)
    #
    # head = FCOSHead(20)
    # outs = head(ps)
    # assert len(outs) == 3
    # for out in outs:
    #     assert len(out) == 5
    #     for out_per_level in out:
    #         print(out_per_level.shape)

    fcos = FCOS()
    preds = fcos(imgs)
    assert len(preds) == 3
    for pred in preds:
        assert len(pred) == 5
        for pred_per_level in pred:
            print(pred_per_level.shape)
    print('-' * 30, '\n')

    loss_func = FCOSLossEvaluator()
    objs = [
        torch.tensor([[10, 20, 150, 280, 1], [35, 50, 100, 200, 10], [60, 55, 180, 135, 15]]),
        torch.tensor([[5, 7, 95, 87, 6], [50, 80, 300, 400, 7]])
    ]
    cls_loss, reg_loss, cnt_loss = loss_func(preds, objs)
    print(f"Classification Loss:{cls_loss:.4f}\nRegression Loss:{reg_loss:.4f}\ncnt_loss:{cnt_loss:.4f}\n")
    print('-' * 30, '\n')

    torch.cuda.empty_cache()

    with torch.no_grad():
        postprocess = FCOSPostProcessor()
        detections = postprocess(*preds, img_sizes)
        print(len(detections))
        for det_per_img in detections:
            print(det_per_img.shape)
            if not det_per_img.numel():
                print("no objects detected")
                continue

            boxes = det_per_img[:, :4].cpu().int().numpy()
            labels, scores = det_per_img[:, -2].cpu().long().numpy(), det_per_img[:, -1].cpu().numpy()

            for label, score, box in zip(labels, scores, boxes):
                print(f"label:{label} score:{score} box:{box}")
            print('-' * 60, '\n')

"""
torch.Size([2, 20, 40, 60])
torch.Size([2, 20, 20, 30])
torch.Size([2, 20, 10, 15])
torch.Size([2, 20, 5, 8])
torch.Size([2, 20, 3, 4])
torch.Size([2, 4, 40, 60])
torch.Size([2, 4, 20, 30])
torch.Size([2, 4, 10, 15])
torch.Size([2, 4, 5, 8])
torch.Size([2, 4, 3, 4])
torch.Size([2, 1, 40, 60])
torch.Size([2, 1, 20, 30])
torch.Size([2, 1, 10, 15])
torch.Size([2, 1, 5, 8])
torch.Size([2, 1, 3, 4])
------------------------------ 

Classification Loss:1.1328
Regression Loss:1.0005
cnt_loss:0.6838

------------------------------ 

2
torch.Size([14, 6])
label:2 score:0.679158627986908 box:[217  42 222  42]
label:2 score:0.6661441326141357 box:[218 147 220 146]
label:2 score:0.6557060480117798 box:[218  18 221  18]
label:2 score:0.6410477161407471 box:[218 169 222 170]
label:2 score:0.6384167075157166 box:[219   3 220   3]
label:2 score:0.6316291689872742 box:[217  26 219  26]
label:2 score:0.6204960346221924 box:[218 194 220 195]
label:2 score:0.610692024230957 box:[217  50 221  50]
label:2 score:0.6096770763397217 box:[218  82 220  82]
label:2 score:0.6078450679779053 box:[216 178 221 178]
label:2 score:0.6064798831939697 box:[218  10 221  10]
label:2 score:0.6003952026367188 box:[218  59 221  58]
label:2 score:0.5969030857086182 box:[216  34 221  34]
label:2 score:0.5957903265953064 box:[217  67 220  67]
------------------------------------------------------------ 

torch.Size([1, 6])
label:18 score:0.665486216545105 box:[318  62 321  62]
------------------------------------------------------------ 
"""