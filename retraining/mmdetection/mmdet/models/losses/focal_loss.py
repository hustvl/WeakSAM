# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from mmdet.registry import MODELS
from .utils import weight_reduce_loss


# This method is only for debugging
def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None,
                          drop = True):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    ############## percentage
    if drop:
        partition = 0.1
        factor = 0
        loss_copy = loss.detach()
        loss_copy = loss_copy.reshape(-1)
        loss_seq, ind = torch.sort(loss_copy, descending=True)
        length = len(loss_seq) * partition
        length = 0 if length <= 1 else int(length)
        std = loss_seq[length]
        indices = torch.where(loss > std)
        loss[indices] *= factor
    ##############
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def py_focal_loss_with_prob(pred,
                            target,
                            weight=None,
                            gamma=2.0,
                            alpha=0.25,
                            reduction='mean',
                            avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.
    Different from `py_sigmoid_focal_loss`, this function accepts probability
    as input.

    Args:
        pred (torch.Tensor): The prediction probability with shape (N, C),
            C is the number of classes.
        target (torch.Tensor): The learning label of the prediction.
            The target shape support (N,C) or (N,), (N,C) means one-hot form.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if pred.dim() != target.dim():
        num_classes = pred.size(1)
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]

    target = target.type_as(pred)
    pt = (1 - pred) * target + pred * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean',
                       drop = True, 
                       avg_factor=None,
                       ignore_index=-100,
                       pos = 0):
    r"""A wrapper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    factor = 0
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(), gamma,
                               alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
        
    # if drop:
    #     partition = 0.1
    #     loss_copy = loss.detach()
    #     loss_copy = torch.sum(loss_copy, dim=1)
    #     fg = torch.where(target != 20)
    #     loss_fg = loss_copy[fg]
    #     loss_fg_seq, seq = torch.sort(loss_fg, descending=True)
    #     if loss_fg_seq.shape == 0:
    #         loss = weight_reduce_loss(
    #                 loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    #         return loss
    #     std_len = int(len(loss_fg) * partition)
    #     std = loss_fg_seq[std_len]
    #     indi = torch.where(loss_fg > std)
    #     indi = torch.unique(indi[0])
    #     if indi == torch.Size([]):
    #         indices = fg[indi]
    #         loss[indices] *= factor
        
    #     loss = weight_reduce_loss(
    #     loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    #     return loss, indi

        
    #     return loss, indi
    # loss = loss.sum()
    # loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    
    
    
    # The default value of ignore_index is the same as F.cross_entropy
    # ignore_index = -100 if ignore_index is None else ignore_index
    # # element-wise losses
    # print(pred.shape)
    # print(max(target))
    # loss = F.cross_entropy(
    #     pred,
    #     target,
    #     weight=None,
    #     reduction='none',
    #     ignore_index=ignore_index)

    # # average loss over non-ignored elements
    # # pytorch's official cross_entropy average loss over non-ignored elements
    # # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa



    # if drop:
    #     # Preliminary stage of sigmoid focal.
    #     pred_sigmoid = pred.sigmoid()
    #     target = target.type_as(pred)
    #     fg = torch.where(target != 20)
    #     tar = F.one_hot(target.to(torch.int64), num_classes = -1).float()[:, :-1]
    #     pt = (1 - pred_sigmoid) * tar + pred_sigmoid * (1 - tar)
    #     loss_ = F.binary_cross_entropy_with_logits(
    #         pred, tar, reduction='none')
    #     loss_ = loss_.sum(dim=1)
    #     part = 0.01
    #     loss_d = loss_.detach()
    #     loss_fg = loss_[fg].detach()
    #     minval = min(loss_d)
    #     minival = minval.clone()
    #     loss_d[fg] = minival
    #     #
    #     loss_d = (loss_d - min(loss_d)) / (max(loss_d) - min(loss_d))
    #     loss_fg = (loss_fg - min(loss_fg)) / (max(loss_fg) - min(loss_d))
    #     #
    #     loss_d[fg] = loss_fg
    #     loss_d_seq, seq = torch.sort(loss_d, descending=True)
    #     leng = int(len(loss_d_seq) * part)
    #     std = loss_d_seq[leng]
    #     indi = torch.where(loss_d > std)
    #     loss[indi] *= factor



    # pred_sigmoid = pred.sigmoid()
    # target = target.type_as(pred)
    # # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    # pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    # focal_weight = (alpha * target + (1 - alpha) *
    #                 (1 - target)) * pt.pow(gamma)
    # loss = F.binary_cross_entropy_with_logits(
    #     pred, target, reduction='none') * focal_weight
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss, -1


@MODELS.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0,
                 activated=False,):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_

        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float, optional): A balanced form for Focal Loss.
                Defaults to 0.25.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            activated (bool, optional): Whether the input is activated.
                If True, it means the input has been activated and can be
                treated as probabilities. Else, it should be treated as logits.
                Defaults to False.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.activated = activated

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                drop = True,
                reduction_override=None,
                pos = 0):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning label of the prediction.
                The target shape support (N,C) or (N,), (N,C) means
                one-hot form.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            if self.activated:
                calculate_loss_func = py_focal_loss_with_prob
            else:
                if pred.dim() == target.dim():
                    # this means that target is already in One-Hot form.
                    calculate_loss_func = py_sigmoid_focal_loss
                elif torch.cuda.is_available() and pred.is_cuda:
                    calculate_loss_func = sigmoid_focal_loss
                else:
                    num_classes = pred.size(1)
                    target = F.one_hot(target, num_classes=num_classes + 1)
                    target = target[:, :num_classes]
                    calculate_loss_func = py_sigmoid_focal_loss

            loss_cls, indi = calculate_loss_func(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor,
                drop=drop,
                pos=pos)
            
            loss_cls *= self.loss_weight
            
            # loss_cls = self.loss_weight * calculate_loss_func(
            #     pred,
            #     target,
            #     weight,
            #     gamma=self.gamma,
            #     alpha=self.alpha,
            #     reduction=reduction,
            #     avg_factor=avg_factor,
            #     drop=drop,
            #     pos=pos)

        else:
            raise NotImplementedError
        return loss_cls, indi
