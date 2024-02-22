import numpy as np 
import os 
from pathlib import Path

from pycocotools.coco import COCO

import cv2


source = 'WeakSAM/WSOD2/data/coco/coco2014/annotations/instances_val2014.json' #json文件的路径
folder = 'WeakSAM/WSOD2/data/coco/coco2014/val2014' # 放图像的文件夹
coco = COCO(source)#实例化
print('Ready!')
records = coco.get_img_ids() #获取所有图像的唯一id，结果是一个list格式
dict2write = {}
for index in range(len(records)):
    cur_array = np.zeros(90)
    record = records[index] #获取其中一个记录即图像id
    img_dicc = coco.loadImgs([record]) #结果是一个list,其元素是字典：
    annIds = coco.getAnnIds(imgIds= record) #获取图像 ID 对应的标注框id
    # print(annIds)
    anns_dicc = coco.loadAnns(annIds)
    for item in anns_dicc:
        # print(item['category_id'])
        cur_array[item['category_id'] - 1] = 1
    cur_array = cur_array.astype(np.uint8)
    dict2write[str(record).rjust(12, '0')] = cur_array
    if index % 1000 == 0:
        print(index, ' images finished.')
np.save('label_cls_val.npy', dict2write)
'''
{
    'file_name': '59e2513a3c4fdb25ff.png',
    'height': 1024,
    'width': 1024,
    'id': 2021000001
}
'''

 # 标注框字典构成的list,每个list元素是一个字典：

'''
anns_dicc[0] ={
'area': 5776,
'bbox': [1, 0, 76, 76],
'category_id': 1,
'id': 1,
'ignore': 0,
'image_id': 2021000001,
'iscrowd': 0,
'segmentation': []
}
'''