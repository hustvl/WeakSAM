import torch
import torch.nn as nn
from torch.nn import functional as F
from wetectron.utils.utils import generate_img_label

class Compute_Cam_Loss(nn.Module):
    def __init__(self, config, channel_size):
        super(Compute_Cam_Loss, self).__init__()
        self.channel_size= channel_size
        self.num_classes = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.fc_cam = nn.Linear(self.channel_size, self.num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature_maps, targets):
        device = feature_maps[0].device
        avg_feats = feature_maps[0].mean(3).mean(2)
        loss_cam = 0


        cam_weight = self.fc_cam.weight.unsqueeze(-1).unsqueeze(-1)
        #class_logits = self.fc_cam(avg_feats)

        for avg_feat, targets_per_im in zip(avg_feats, targets):
            labels_per_im = targets_per_im.get_field('labels').unique()
            labels_per_im = generate_img_label(self.num_classes, labels_per_im, device)

            cam_logits = self.fc_cam(avg_feat)
            loss_cam += F.binary_cross_entropy_with_logits(cam_logits, labels_per_im.clamp(0, 1))

        loss_cam /= len(targets)

        atten_logits = nn.functional.conv2d(feature_maps[0], weight=cam_weight, bias=None)[:,1:]
        atten_map = torch.sigmoid(torch.mean(atten_logits,dim=1))

        #if truncate:
        #    weight = torch.where(torch.gt(layer.weight, 0.),
        #                         layer.weight, torch.zeros_like(layer.weight))
        #    cam_loss = F.binary_cross_entropy_with_logits(, labels_per_im.clamp(0, 1))

        return atten_logits, atten_map, loss_cam
