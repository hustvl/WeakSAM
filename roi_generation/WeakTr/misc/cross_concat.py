import os
import numpy 
import sys
from pathlib import Path

cross_path = '/home/junweizhou/WeakTr/WeakTr/weaktr_results/VOC12-peak-test/cross'
peakfile_list = os.listdir(cross_path)

saving_name = 'peak-pam-k129-t90'
saving_path_val = os.path.join(cross_path, saving_name)
saving_path = os.path.join(cross_path, saving_name)
Path(saving_path).mkdir(exist_ok=True, parents=True)
Path(saving_path_val).mkdir(exist_ok=True, parents=True)

for peakfile in peakfile_list:
    if 'val' in peakfile:
        pth = os.path.join(cross_path, peakfile)
        val_list = os.listdir(pth)
        for item in val_list:
            file = open(os.path.join(saving_path_val, item), 'a', encoding='utf-8')
            for line in open(os.path.join(pth, item)):
                file.writelines(line)
            file.close()
    else:
        pth = os.path.join(cross_path, peakfile)
        train_list = os.listdir(pth)
        for item in train_list:
            file = open(os.path.join(saving_path, item), 'a', encoding='utf-8')
            for line in open(os.path.join(pth, item)):
                file.writelines(line)
            file.close()