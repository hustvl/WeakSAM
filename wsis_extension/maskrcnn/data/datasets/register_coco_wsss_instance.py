from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
import os
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

_PREDEFINED_SPLITS_COCO = {"coco":
    {"coco_2017_train_wsss": ("coco/train2017", "coco/annotations/instances_train2017_wsss.json")}
}

def register_all_coco(root):
    for dataset_name, splits_per_dataset in _PREDEFINED_SPLITS_COCO.items():
        for key, (image_root, json_file) in splits_per_dataset.items():
            # Assume pre-defined datasets live in `./datasets`.
            register_coco_instances(
                key,
                _get_builtin_metadata(dataset_name),
                os.path.join(root, json_file) if "://" not in json_file else json_file,
                os.path.join(root, image_root),
            )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_coco(_root)
