import torch
import torch.nn as nn
from functools import partial
from vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

import math

from AAF import AAF

__all__ = [
    'deit_small_WeakTr_patch16_224',
]


class WeakTr(VisionTransformer):
    def __init__(self, depth=12, num_heads=6, reduction=4, pool="avg", *args, **kwargs):
        super().__init__(depth=depth, num_heads=num_heads, *args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_classes, self.embed_dim))

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        print(self.training)

        self.adaptive_attention_fusion = AAF(depth * num_heads, reduction, pool)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.pos_embed.shape[1] - self.num_classes
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        # first through patch embedding and get attention maps of origin shape.
        # with BCWH being the shape of input image.(Noticing that BCWH!)
        

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # The providing N^2 + C
        x = x + self.interpolate_pos_encoding(x, w, h) # adding positional encoding.
        x = self.pos_drop(x)
        attn_weights = []

        # adding learnable weights.
        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)
        # x: self-attn map inside the bl
        return x[:, 0:self.num_classes], x[:, self.num_classes:], attn_weights

    def forward(self, x, return_att=False, attention_type='fused'):
        w, h = x.shape[2:]

        x_cls, x_patch, attn_weights = self.forward_features(x)
        # In general, x is exactly the feature embeddings(feat_map) that we want in the paper.
        ###########################
        # TODO@: SAVING X FOR AUGMENTATION OF PEAK GENERATION. each self attn map is of practival use?
        # obtaining features from attention layers. See the func
        n, p, c = x_patch.shape
        if w != h:  # standard input 224 * 224
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p ** 0.5), int(p ** 0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])  # rearranging.
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)   # Conv Head
        coarse_cam_pred = self.avgpool(x_patch).squeeze(3).squeeze(2)


#### attention weights for the cross attention generation.
        attn_weights = torch.stack(attn_weights)  # 12 * B * H * N * N
        # attn_weights representing learnable nodes for attention map fusion.

        attn_weights_detach = attn_weights.detach().clone()
        k, b, h, n, m = attn_weights_detach.shape
        attn_weights_detach = attn_weights_detach.permute([1, 2, 0, 3, 4]).contiguous()
        attn_weights_saving = attn_weights_detach.detach().clone()
        
        # attention weights: b num_heads layer_depth n n (DeiT small with 6 heads and 12 layers.)
        
        attn_weights_detach = attn_weights_detach.view(b, h * k, n, m)
        
        weighted_attn_maps = self.adaptive_attention_fusion(attn_weights_detach)
        weighted_attn_maps = weighted_attn_maps.view(b, h, k, n, m)
        head = h

        coarse_cam = x_patch.detach().clone()  # B * C * 14 * 14
        coarse_cam = F.relu(coarse_cam)
        # see here for the coarse cam.
        n, c, h, w = coarse_cam.shape
        
        attn_weights_saving = attn_weights_saving[:, :, :, 0:self.num_classes, self.num_classes:].reshape([n, head, k, c, h, w])
        cross_attn = weighted_attn_maps.mean(1).mean(1)[:, 0:self.num_classes, self.num_classes:].reshape([n, c, h, w])

        # cross attention for peak extraction.##########################
        #TODO@: SAVING CROSS ATTN MAPS OF DIFFERENT HEADS TO REPLACE SAM BASE IN THE PROPOSAL.
        
        if attention_type == 'fused':
            cams = cross_attn * coarse_cam  # B * C * 14 * 14
        elif attention_type == 'patchcam':
            cams = coarse_cam
        else:
            cams = cross_attn

        patch_attn = weighted_attn_maps.mean(1).mean(1)[:, self.num_classes:, self.num_classes:]

        fine_cam = torch.matmul(patch_attn.unsqueeze(1), cams.view(cams.shape[0],
                                                                        cams.shape[1], -1, 1)). \
            reshape(cams.shape[0], cams.shape[1], h, w)

        fine_cam_pred = self.avgpool(fine_cam).squeeze(3).squeeze(2)

        patch_attn = patch_attn.unsqueeze(0)

        cls_token_pred = x_cls.mean(-1)

        if return_att:  # api here for CAM generation.
            return cls_token_pred, cams, patch_attn, coarse_cam, cross_attn, attn_weights_saving
        # COARSE CAM AND CROSS ATTN ADDED BY COLEZ.
        else:
            return cls_token_pred, coarse_cam_pred, fine_cam_pred


@register_model
def deit_small_WeakTr_patch16_224(pretrained=False, **kwargs):
    model = WeakTr(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model
