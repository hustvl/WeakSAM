import numpy as np
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import xml.dom.minidom as md

pth = '/home/junweizhou/WeakTr/WeakTr/voc12/cls_labels1.npy'
temp = np.load(pth, allow_pickle=True)
print(temp.item()['2008_000021'])

categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

path = '/home/junweizhou/WeakTr/WeakTr/data/voc12/VOCdevkit/VOC2012/Annotations'
txt_file = '/home/junweizhou/WeakTr/WeakTr/data/voc12/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'

data2write = dict()
f = open(txt_file, encoding='utf-8')
num_list = []
for line in f:
    num_list.append(line.strip())

for num in num_list:
    if num not in data2write.keys():
        anno = path + '/' + num + '.xml'
        tree = ET.parse(anno)
        root = tree.getroot()
        label = np.zeros(20).astype(np.float32)
        for object in root.findall('object'):
            object_name = object.find('name').text
            idx = categories.index(object_name)
            label[idx] = 1
        data2write[num] = label


np.save('./voc12/cls_labels1.npy', data2write)