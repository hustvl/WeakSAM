from ..builder import DETECTORS
from .two_stage import TwoStageDetector


@DETECTORS.register_module()
class WeakRCNN(TwoStageDetector):
    """Implementation of `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 pretrained=None):
        super(WeakRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_labels,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)

        losses = dict()

        proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_labels, **kwargs)
        losses.update(roi_losses)
        return losses


    def forward_test(self, imgs, img_metas, proposals, **kwargs):
        # whole api here.
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], proposals[0],
                                    **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            return self.aug_test(imgs, img_metas, proposals, **kwargs)


    def aug_test(self, imgs, img_metas, proposal_list, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)
