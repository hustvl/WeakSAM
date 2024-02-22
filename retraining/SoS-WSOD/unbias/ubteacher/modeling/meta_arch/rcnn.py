# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances, ROIMasks

@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        # print(images.image_sizes)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        # unsupervised data weak 指的是 teachernet 生成 pgt, 因此是不需要计算 loss 的
        elif branch == "unsup_data_weak":
            # Region proposal network
            # 输入的 None 是原本 gt_instances, 这里是没有的
            # 输出第二项是 loss 是空的, 这里不需要
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # 需要处理一次啊 pred_mask 变成 正常的 bit_mask 才成
            if proposals_roih[0].has("pred_masks"): 
                """
                proposals_roih 每项中 _fields 包含:
                    pred_boxes, scores, pred_classes, pred_masks
                另外有 _image_size
                """
                # print("rcnn: ",proposals_roih[0]._fields.keys())
                # print("rcnn: ", type(proposals_roih[0]["pred_masks"]))
                # print("rcnn: ", proposals_roih[0].pred_masks.shape)
                # exit()
                # 这里直接执行 postprocess 里相关的操作来变换 pred_masks
                processed_results = []
                # TODO: mask_threshold=0.5 其实应该设置成超参数的
                for results_per_image in proposals_roih:
                    height, width = results_per_image.image_size
                    roi_masks = ROIMasks(results_per_image.pred_masks[:, 0, :, :])
                    results_per_image.pred_masks = roi_masks.to_bitmasks(
                        results_per_image.pred_boxes, height, width, 0.5
                    ).tensor
                    processed_results.append(results_per_image)
                proposals_roih = processed_results

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        # 用 ublabel data 进行训练的时候不添加定位 loss
        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

    # def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
    #     assert not self.training

    #     images = self.preprocess_image(batched_inputs)
    #     features = self.backbone(images.tensor)

    #     if detected_instances is None:
    #         if self.proposal_generator:
    #             proposals, _ = self.proposal_generator(images, features, None)
    #         else:
    #             assert "proposals" in batched_inputs[0]
    #             proposals = [x["proposals"].to(self.device) for x in batched_inputs]

    #         results, _ = self.roi_heads(images, features, proposals, None)
    #     else:
    #         detected_instances = [x.to(self.device) for x in detected_instances]
    #         results = self.roi_heads.forward_with_given_boxes(
    #             features, detected_instances
    #         )

    #     if do_postprocess:
    #         return GeneralizedRCNN._postprocess(
    #             results, batched_inputs, images.image_sizes
    #         )
    #     else:
    #         return results
