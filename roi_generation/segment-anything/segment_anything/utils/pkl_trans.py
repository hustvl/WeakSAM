import pickle
import argparse
import numpy as np
import os 
from pycocotools.coco import COCO

def index_list(ann_path, split, ):
    numpath = os.path.join(ann_path, str(split) + '.txt')
    
    f = open(numpath, encoding='utf-8')
    num_list = []
    for line in f:
        num_list.append(line.strip())
    numlist = []
    for num in num_list:
        num1 = num.replace('_', '')
        numlist.append(int(num1))
    return numlist
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='voc', help='dataset type from voc, coco')
    parser.add_argument('--image-set', default='trainval' )
    parser.add_argument('--ann-path', default='/home/junweizhou/WeakSAM/WSOD2/data/voc/VOC2007/ImageSets/Main')
    parser.add_argument('--tar-proposals', default='/home/junweizhou/WeakSAM/segment-anything/peak_proposal/VOC07/grid32_fine.pkl')
    parser.add_argument('--save-path', default='/home/junweizhou/WeakSAM/MIST/proposal/SAM/VOC07')
    parser.add_argument('--save-name', default='voc07-sam_fineboxes.pkl')
    args =  parser.parse_args()
    
    dataset = args.dataset
    split = args.image_set
    ann_path = args.ann_path
    tar_prop = args.tar_proposals
    save_path = args.save_path
    save_name = args.save_name
    
    if dataset == 'voc':
        numlist = index_list(ann_path, split)
    elif dataset == 'coco':
        cocoann = COCO(ann_path)
        imgids = cocoann.getImgIds()
        imgids.sort()
        numlist = imgids
    
    ori = {}
    fi = tar_prop
    f = open(fi, 'rb')
    data = pickle.load(f)
    data2write = []
    label2write = []
    for datum in data:
        datum = datum.astype(np.uint16)
        lab_len = len(datum)
        label = np.ones(lab_len).astype(np.float32)
        if dataset == 'coco':
            label = label.reshape(lab_len, 1)
            datum = datum.astype(np.float32)
        data2write.append(datum)
        label2write.append(label)

    ori['boxes'] = data2write
    ori['scores'] = label2write 
    ori['indexes'] = numlist
    
    whole_pth = os.path.join(save_path, save_name)
    with open(whole_pth, 'wb') as file:
        pickle.dump(ori, file)