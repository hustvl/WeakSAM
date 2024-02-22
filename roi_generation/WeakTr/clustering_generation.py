import os
import numpy as np 
import sklearn
import argparse
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.cluster import OPTICS
from sklearn.cluster import KMeans
from pathlib import Path

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.0)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
    
def batch_readtxt(folder_path, img_label, starting_layernum=1, ending_layernum=12):
    whole_pth = os.path.join(folder_path, str(img_label) + '.txt')
    if os.path.getsize(whole_pth) == 0:
        return None, None
    pt_list = np.genfromtxt(whole_pth, dtype=[float, float, int, float, int], delimiter=' ') # the read all txt
    sorted_pts = {}

    
    pts_list = []
    if pt_list.size == 1:
        #point all
        x = pt_list.item()[0]
        y = pt_list.item()[1]
        pts_list.append([x, y])
        # cls sorted
        cls = pt_list.item()[2]
        if cls not in sorted_pts:
            sorted_pts[cls] = []
            sorted_pts[cls].append([x, y])
        return pts_list, sorted_pts

    for item in pt_list:
        x = item[0]
        y = item[1]
        head_idx = item[4]
        if (head_idx % 12) + 1 < starting_layernum or head_idx % 12 + 1 > ending_layernum:  # confining starting and ending visualization.
            continue
        item_ = [x, y]
        pts_list.append(item_)
        cls = item[2]
        #####sorted points
        if cls not in sorted_pts:
            sorted_pts[cls] = []
            sorted_pts[cls].append(item_)
        else:
            sorted_pts[cls].append(item_)
    return pts_list, sorted_pts
    
    
    
if __name__ == '__main__':
    parser  = argparse.ArgumentParser()
    parser.add_argument('--image-set', default=None)
    parser.add_argument('--image-path', default=None)
    parser.add_argument('--saving-path', default='./vis')
    parser.add_argument('--point-path', default=None)
    parser.add_argument('--start', default=1, type=int)
    parser.add_argument('--end', default=12, type=int)
    parser.add_argument('--iscoco', default=False)
    parser.add_argument('--gamma', default=0.3, type=float)
    parser.add_argument('--peakfile', default=None, help='The path to store clustered peak points.')
    
    args = parser.parse_args()
    peak_path = args.peakfile
    Path(peak_path).mkdir(exist_ok=True, parents=True)
    gamma = args.gamma
    f = open(args.image_set, encoding='utf-8')
    num_list = []
    for line in f:
        if args.iscoco:
            if 'val' in args.image_path:
                num_list.append('COCO_val2014_' + line.strip())
            else:
                num_list.append('COCO_train2014_' + line.strip())
        else:
            num_list.append(line.strip())
    
    cnt = 0
    for i, num in enumerate(num_list):
        # print(num)
        if i > -1: 
            if i % 500 == 0:
                print('%d image processed.' % i)
            peak_txt = open(os.path.join(peak_path, "%s.txt" % num), 'w')
            points, sorted_points  = batch_readtxt(args.point_path, num)
            if points is None:
                peak_txt.write("%d %d %d %.3f\n" % (100, 100, 0, 1.000))
                continue
            if len(points) // 5 == 0:
                clst = 1
            else:
                clst = len(points) // 5
            clustering = KMeans(n_clusters=clst, random_state=0, n_init='auto').fit(points)
            centroids = clustering.cluster_centers_
            cnt += len(centroids)
            for centroid in centroids:
                peak_txt.write("%d %d %d %.3f\n" % (centroid[0], centroid[1], 0, 1.000))
        
        else :
            points, sorted_points  = batch_readtxt(args.point_path, num)
            clustering = KMeans(n_clusters=len(points) // 5, random_state=0, n_init='auto').fit(points)
            centroids = clustering.cluster_centers_
            cnt += len(centroids)
            ######### For OPTICS Clustering 
            # pts = []
            # for item in sorted_points.values():
            #     item = np.array(item)
                # item1 = []
                # clustering = OPTICS(min_samples=3).fit(item)
                # indices = np.where(clustering.labels_ == -1)
                # scattered = item[indices]
                # for pt in scattered:
                #     pts.append(pt)
                #     item1.append(pt)
                #     for points in pts[: -1]:
                #         if (pt[0] - points[0]) ** 2 + (pt[1] - points[1]) ** 2 < 225:
                #             points[0] = (1 - gamma) * pt[0] + gamma * points[0]
                #             points[1] = (1 - gamma) * pt[1] + gamma * points[1]
                #             del item1[-1]
                #             del pts[-1] 
                #             break
                #########
                
                        
                # for i in range(clustering.labels_.max()):
                #     indices = np.where(clustering.labels_ == i)
                #     single_cluster = item[indices]
                #     single_cluster = single_cluster.mean(0)
                #     if pts == []:
                #         pts.append(single_cluster)
                #         item1.append(single_cluster)
                #     else:
                #         item1.append(single_cluster)
                #         pts.append(single_cluster)
                #         for points in pts[: -1]:
                #             if (single_cluster[0] - points[0]) ** 2 + (single_cluster[1] - points[1]) ** 2 < 900:
                #                 points[0] = (single_cluster[0] + points[0]) // 2
                #                 points[1] = (single_cluster[1] + points[1]) // 2
                #                 del item1[-1]
                #                 del pts[-1] 
                #                 break        
            # pts = np.array(pts)
            pts = np.array(centroids)
            points = np.array(points)
            Path(args.saving_path).mkdir(parents=True, exist_ok=True)
            imgpth = os.path.join(args.image_path, str(num) + '.jpg')
            image = cv2.imread(imgpth)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            # show_mask(masks[0], plt.gca())
            # show_box(gts[i], plt.gca(), color='red', lw=2)
            show_points(pts, np.ones(pts.shape[0]), plt.gca())
            plt.axis('off')
            plt.savefig(args.saving_path + '/' + str(num))
            plt.close()
    
    cnt /= len(num_list)
    print('average centroids number:', cnt)