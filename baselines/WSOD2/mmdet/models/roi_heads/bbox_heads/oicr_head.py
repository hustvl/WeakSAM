import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, WeaklyMulticlassNMS, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

@HEADS.register_module()
class OICRHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 hidden_channels=1024,
                 bbox_coder=None,
                 num_classes=20):
        super(OICRHead, self).__init__()
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fp16_enabled = False

        in_channels *= self.roi_feat_area

        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(hidden_channels, hidden_channels)
        self.dropout2 = nn.Dropout()

        self.fc_cls1 = nn.Linear(hidden_channels, num_classes)
        self.fc_cls2 = nn.Linear(hidden_channels, num_classes)

        self.fc_refine1 = nn.Linear(hidden_channels, num_classes + 1)
        self.fc_refine2 = nn.Linear(hidden_channels, num_classes + 1)
        self.fc_refine3 = nn.Linear(hidden_channels, num_classes + 1)

        self.with_bbox = False
        if bbox_coder is not None:
            self.fc_bbox = nn.Linear(hidden_channels, 4)
            self.bbox_coder = build_bbox_coder(bbox_coder)
            self.with_bbox = True

        self.weakly_multiclass_nms = WeaklyMulticlassNMS(num_classes)

    def init_weights(self):
        nn.init.normal_(self.fc_cls1.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls1.bias, 0)
        nn.init.normal_(self.fc_cls2.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls2.bias, 0)
        nn.init.normal_(self.fc_refine1.weight, 0, 0.01)
        nn.init.constant_(self.fc_refine1.bias, 0)
        nn.init.normal_(self.fc_refine2.weight, 0, 0.01)
        nn.init.constant_(self.fc_refine2.bias, 0)
        nn.init.normal_(self.fc_refine3.weight, 0, 0.01)
        nn.init.constant_(self.fc_refine3.bias, 0)
        if self.with_bbox:
            nn.init.normal_(self.fc_bbox.weight, 0, 0.001)
            nn.init.constant_(self.fc_bbox.bias, 0)

    @auto_fp16()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        cls1 = self.fc_cls1(x)
        cls2 = self.fc_cls2(x)

        cls1 = F.softmax(cls1, dim=1)
        cls2 = F.softmax(cls2, dim=0)
        cls = cls1 * cls2

        refine1 = F.softmax(self.fc_refine1(x), dim=1)
        refine2 = F.softmax(self.fc_refine2(x), dim=1)
        refine3 = F.softmax(self.fc_refine3(x), dim=1)

        if self.with_bbox:
            bbox = self.fc_bbox(x)
        else:
            bbox = None

        return cls, refine1, refine2, refine3, bbox

    @force_fp32(apply_to=('cls'))
    def loss_wsddn(self, cls, labels):
        cls = cls.sum(dim=0)
        cls = torch.clamp(cls, 0., 1.)
        loss_wsddn = F.binary_cross_entropy(cls, labels.float(), reduction='sum')
        return loss_wsddn

    @force_fp32(apply_to=('cls'))
    def loss_oicr(self, cls, labels, weights):
        labels += 1
        labels = F.one_hot(labels, self.num_classes + 1)
        #loss_oicr = F.cross_entropy(cls, labels, reduction='none')
        loss_oicr = (- labels * (cls + 1e-6).log()).sum(dim=1)
        loss_oicr = (loss_oicr * weights).mean()
        return loss_oicr

    @force_fp32(apply_to=('bbox'))
    def loss_bbox(self, bbox, targets, labels, weights):
        pos_idx = labels.nonzero().squeeze(1)
        bbox = bbox[pos_idx]
        targets = targets[pos_idx]
        weights = weights[pos_idx]

        loss_bbox = F.smooth_l1_loss(bbox, targets, reduction='none')
        weights = weights.view(-1, 1)
        loss_bbox = 30 * (loss_bbox * weights).mean()
        return loss_bbox


    @force_fp32(apply_to=('bbox')) 
    def get_targets(self, bbox, gt):
        bbox_targets = self.bbox_coder.encode(bbox, gt)
        return bbox_targets


    @force_fp32(apply_to=('cls'))
    def get_bboxes(self,
                   rois,
                   scores,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        scores_pad = torch.zeros((scores.shape[0], 1), dtype=torch.float32).to(device=scores.device)
        scores = torch.cat([scores, scores_pad], dim=1) # paddding to the same shape.

        if bbox_pred is not None:  # through this.
            bboxes = self.bbox_coder.decode(
                rois[:, 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1])
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0])

        if rescale and bboxes.size(0) > 0:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                scale_factor = bboxes.new_tensor(scale_factor)
                bboxes = (bboxes.view(bboxes.size(0), -1, 4) /
                          scale_factor).view(bboxes.size()[0], -1)

        if cfg is None:
            return bboxes, scores
        else:
            #det_bboxes, det_labels = self.weakly_multiclass_nms(bboxes, scores, cfg.nms, cfg.max_per_img)
            det_bboxes, det_labels = multiclass_nms(bboxes, scores, cfg.score_thr, cfg.nms, cfg.max_per_img)

            return det_bboxes, det_labels


