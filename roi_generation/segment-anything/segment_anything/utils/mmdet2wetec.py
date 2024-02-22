import pickle as pkl
import os
import numpy as np
import torch
from pathlib import Path
import argparse

# path1 = '/home/junweizhou/WSOD/OD-WSCL/proposal/SAM/VOC12/SS-voc_2012_train-boxes.pkl'
# path = '/home/junweizhou/WSOD/OD-WSCL/proposal/SAM/MCG-coco_2014_train-boxes.pkl'
# with open(path, 'rb') as f:
#     proposals = pkl.load(f, encoding='latin1')

# with open(path1, 'rb') as f:
#     proposals1 = pkl.load(f, encoding='latin1')

# new_idx = proposals['indexes']

# path2replace = '/home/junweizhou/WeakSAM/segment-anything/peak_proposal/COCO14/cocotrain.pkl'
# with open(path2replace, 'rb') as f:
#     newp = pkl.load(f, encoding='latin1')
# proposals['boxes'] = newp
# path_storage = '/home/junweizhou/WSOD/OD-WSCL/proposal/SAM/COCO'
# file_name = 'coco-trainboxes.pkl'
# Path(path_storage + '/').mkdir(exist_ok=True, parents=True)
# os.chdir(path_storage + '/')
# with open(file_name, 'wb') as f:
#     pkl.dump(proposals, f)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-proposal', default=None, help='mmdet format SAM proposal set')
    parser.add_argument('--target-proposal', default=None, help='The original proposal set, SS & MCG')
    parser.add_argument('--save-name', default=None)
    parser.add_argument('--save-path', default=None)
    args = parser.parse_args()
    
    source_proposal = args.source_proposal
    target_proposal = args.target_proposal
    file_name = args.save_name
    file_path = args.save_path
    
    with open(source_proposal, 'rb') as f:
        proposals_fillin = pkl.load(f, encoding='latin1')
        
        
    with open(target_proposal, 'rb') as f:
        proposals_ori = pkl.load(f, encoding='latin1')
        proposals_ori['boxes'] = proposals_fillin
        
    Path(file_path + '/').mkdir(exist_ok=True, parents=True)
    os.chdir(file_path + '/')
    with open(file_name, 'wb') as f:
        pkl.dump(proposals_ori, f)
        
    
    
    
    