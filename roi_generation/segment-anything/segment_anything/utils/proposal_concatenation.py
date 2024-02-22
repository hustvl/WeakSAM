import os
import sys
import numpy as np
import torch 
import pickle as pkl
import argparse
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cat-pickle-list', default= ['trainval-grid32.pkl', 'coarse/k17_t90.pkl', 'fine/k17_t90.pkl', 'cross/k129_t90.pkl'], type=list
                        , help='receiving a list of pickle format files to be concatenated.')
    parser.add_argument('--pickle-path', default='./peak_proposals/VOC07/', help='path to the pickle files for concatenation.')
    parser.add_argument('--storage-path', default='./peak_proposals/VOC07/', help='the path for the storage of output pickle file.')
    parser.add_argument('--file-name', default='voc07trainval.pkl')
    parser.add_argument('--istest', default=False)

    args = parser.parse_args()
    
    pkl_list = args.cat_pickle_list
    assert isinstance(pkl_list, list), 'The provided pickle list should be a list type.'
    path_pkl = args.pickle_path
    path_storage = args.storage_path
    file_name = args.file_name
    
    data2write = []
    for file in pkl_list:
        if args.istest:
            file_path = os.path.join(path_pkl, file + '_test')
        else:
            file_path = os.path.join(path_pkl, file)
        with open(file_path, 'rb') as f:
            batched_proposal = pkl.load(f)
        if data2write == []:
            data2write = batched_proposal
        else:
            for i in range(len(data2write)):
                if batched_proposal[i].size == 0:
                    continue
                if data2write[i].size == 0:
                    data2write[i] = batched_proposal[i]
                    if data2write[i].size == 0 and i == len(data2write) - 1:
                        print('warning: img %d has no proposal.' %i)
                        ValueError, 'empty img found.'
                print(data2write[i].shape)
                data2write[i] = np.concatenate((data2write[i], batched_proposal[i]), axis=0).astype(np.float32)
    cnt = 0
    for i in range(len(data2write)):
        data2write[i] = data2write[i].astype(np.float32)   
        cnt += data2write[i].shape[0]  
    cnt =  int(cnt / len(data2write))      
    print('average proposal num :' , cnt)
    Path(path_storage + '/').mkdir(exist_ok=True, parents=True)
    os.chdir(path_storage + '/')
    with open(file_name, 'wb') as f:
            pkl.dump(data2write, f)   
    

    
    
