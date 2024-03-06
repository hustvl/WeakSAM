from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
import os

VOC_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "aeroplane"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "bird"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "boat"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "bottle"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "car"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "cat"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "chair"},
    {"color": [0, 0, 192], "isthing": 1, "id": 10, "name": "cow"},
    {"color": [250, 170, 30], "isthing": 1, "id": 11, "name": "diningtable"},
    {"color": [100, 170, 30], "isthing": 1, "id": 12, "name": "dog"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "horse"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "motorbike"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "person"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "pottedplant"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "sheep"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "sofa"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "train"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "tvmonitor"}, ]

_PREDEFINED_SPLITS_VOC12 = {
    "voc12_train_coco_wsss": ("VOC2012/JPEGImages", "VOC2012/voc_2012_train_cocostyle_wsss.json"),
    "voc12_train_coco": ("VOC2012/JPEGImages", "VOC2012/voc_2012_train_cocostyle.json"),
    "voc12_val_coco": ("VOC2012/JPEGImages", "VOC2012/voc_2012_val_cocostyle.json"),
    "voc12_train_aug_coco_wsss": ("VOC2012/JPEGImages", "VOC2012/voc_2012_train_aug_cocostyle_wsss.json"),
}


def _get_voc12_instances_meta():
    thing_ids = [k["id"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 20, len(thing_ids)
    # Mapping from the incontiguous VOC category id to an id in [0, 19]
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in VOC_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret


def register_all_voc12(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_VOC12.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_voc12_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )



_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_voc12(_root)
