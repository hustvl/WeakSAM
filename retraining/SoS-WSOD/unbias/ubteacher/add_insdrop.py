from detectron2.config import CfgNode as CN


def add_insdrop_config(cfg):

# ---------------------------------------------------------------------------- #
# INSTANCE DROP ADDED BY COLEZ.
# ---------------------------------------------------------------------------- #
    _C = cfg
    _C.MODEL.INSTANCE_DROP = CN()
    _C.MODEL.INSTANCE_DROP.ISON = False
    _C.MODEL.INSTANCE_DROP.THRESH_CLS = 0.0
    _C.MODEL.INSTANCE_DROP.THRESH_REG = 0.0
    _C.MODEL.INSTANCE_DROP.RPN_DROP = 0.0
    _C.MODEL.INSTANCE_DROP.OPERATE_ITER = 0