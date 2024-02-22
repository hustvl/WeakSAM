from collections import OrderedDict
from mmdet.evaluation import eval_map, eval_recalls
import os.path as osp
import xml.etree.ElementTree as ET
import numpy as np
from mmcv.utils import print_log

source_path = '/home/junweizhou/Datasets/VOC/VOC2007/'

classes = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
               'tvmonitor')

def cat2label(cls):
    label = {cat: i for i, cat in enumerate(cls)}
    return label



def get_ann(dataset_type, data_path, dataset_year, min_size = None, test_mode = False):
    '''

    :param dataset_type:(str) for the dataset type, (trainval or test)
    :param data_path: (str) the sys path for VOC07  dataset
    :return: a list for all the annotations of the detection task
    '''
    if dataset_year != '2007' or '2012':
        raise KeyError(f'The VOC{dataset_year} dataset does not exists.')
    allowed_types = ['trainval', 'train', 'test']
    if dataset_type not in allowed_types:
        raise KeyError(f'The input dataset type {dataset_type} is not supported.')
    image_folder = data_path + 'VOC' + dataset_year + '/JPEGImages'
    txt_path = data_path + 'VOC' + dataset_year + f'/ImageSets/Layout/{dataset_type}.txt'
    ann_path = data_path + 'VOC' + dataset_year + f'/Annotations'
    f = open(txt_path, encoding='utf-8')
    num_list = []
    for line in f:
        num_list.append(line.strip())
    for ann_num in num_list:
        cur_ann = ann_path + f'/{ann_num}.xml'
#TODO: reading the xml file for annotations.
        tree = ET.parse(cur_ann)
        root = tree.getroot()
        bboxes = []
        labels = []
        bboxes_ignore = []
        labels_ignore = []
        anns_id = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in classes:
                continue
            label = cat2label(classes)[name]
            anns_id.append(ann_num)
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('bndbox')
            bbox = [
                int(float(bnd_box.find('xmin').text)),
                int(float(bnd_box.find('ymin').text)),
                int(float(bnd_box.find('xmax').text)),
                int(float(bnd_box.find('ymax').text))
            ]
            ignore = False

            if min_size:
                assert not test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w < min_size or h < min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
            else:
                bboxes.append(bbox)
                labels.append(label)

        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0,))
        else:
            bboxes = np.array(bboxes, ndmin=2) - 1
            labels = np.array(labels)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0,))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - 1
            labels_ignore = np.array(labels_ignore)

        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            anns_id=np.array(anns_id, dtype=np.int64),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64))
        return ann


def evaluate_voc_protocol(data_path, results,
                 dataset_year,
                 dataset_type = 'trainval',
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=0.5,
                 scale_ranges=None):
    """Evaluate in VOC protocol.

            Args:
                results (list[list | tuple]): Testing results of the dataset.
                metric (str | list[str]): Metrics to be evaluated. Options are
                    'mAP', 'recall'.
                logger (logging.Logger | str, optional): Logger used for printing
                    related information during evaluation. Default: None.
                proposal_nums (Sequence[int]): Proposal number used for evaluating
                    recalls, such as recall@100, recall@1000.
                    Default: (100, 300, 1000).
                iou_thr (float | list[float]): IoU threshold. Default: 0.5.
                scale_ranges (list[tuple], optional): Scale ranges for evaluating
                    mAP. If not specified, all bounding boxes would be included in
                    evaluation. Default: None.

            Returns:
                dict[str, float]: AP/recall metrics.
            """

    if not isinstance(metric, str):
        assert len(metric) == 1
        metric = metric[0]
    allowed_metrics = ['mAP', 'recall']
    if metric not in allowed_metrics:
        raise KeyError(f'metric {metric} is not supported')
    annotations = [get_ann(dataset_type, data_path, dataset_year)] # this annotation is composed by one ndarray
    # a list with k ndarrays required.
    eval_results = OrderedDict()
    iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
    if metric =='mAP':
        assert isinstance(iou_thrs, list)
        if dataset_year == 2007:
            ds_name = 'voc07'
        else:
            ds_name = classes
        mean_aps = []

        #TODO: for the proposal mAP calculation, we the cls should be ignored.
        for iou_thr in iou_thrs:
            print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
            mean_ap, _ = eval_map(
                results,
                annotations,
                scale_ranges=None,
                iou_thr=iou_thr,
                dataset=ds_name,
                logger=logger)