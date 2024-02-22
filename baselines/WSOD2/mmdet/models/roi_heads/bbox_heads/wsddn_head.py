import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, WeaklyMulticlassNMS, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy

@HEADS.register_module()
class WSDDNHead(nn.Module):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 roi_feat_size=7,
                 in_channels=256,
                 hidden_channels=1024,
                 num_classes=20):
        super(WSDDNHead, self).__init__()
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

        self.eps = 1e-5

        #self.weakly_multiclass_nms = WeaklyMulticlassNMS(20)

    def init_weights(self):
        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc_cls1.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls1.bias, 0)
        nn.init.normal_(self.fc_cls2.weight, 0, 0.01)
        nn.init.constant_(self.fc_cls2.bias, 0)

    @auto_fp16()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        cls1 = self.fc_cls1(x)
        cls2 = self.fc_cls2(x)
        return cls1, cls2

    @force_fp32(apply_to=('cls1', 'cls2'))
    def loss(self,
             cls1,
             cls2,
             labels):
        losses = dict()

        cls1 = F.softmax(cls1, dim=1)
        cls2 = F.softmax(cls2, dim=0)

        cls = cls1 * cls2
        cls = cls.sum(dim=0)
        cls = torch.clamp(cls, 0., 1.)

        labels = torch.cat(labels, dim=0)

        loss_wsddn = F.binary_cross_entropy(cls, labels.float(), reduction='sum')

        losses['loss_wsddn'] = loss_wsddn

        return losses

    @force_fp32(apply_to=('cls1', 'cls2'))
    def get_bboxes(self,
                   rois,
                   cls1,
                   cls2,
                   bbox_pred,
                   img_shape,
                   scale_factor,
                   rescale=False,
                   cfg=None):

        cls1 = F.softmax(cls1, dim=1)
        cls2 = F.softmax(cls2, dim=0)

        scores = cls1 * cls2
        scores_pad = torch.zeros((scores.shape[0], 1), dtype=torch.float32).to(device=scores.device)
        scores = torch.cat([scores, scores_pad], dim=1)

        if bbox_pred is not None:
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

