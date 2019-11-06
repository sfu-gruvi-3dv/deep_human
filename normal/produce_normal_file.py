import numpy as np
import sys
sys.path.append('../')
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import util, util_detail
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
from skimage import io
import glob2
from PointCloud import PointCloud
import normal_util
from datetime import datetime

def produce_normal_file(data_dir):
    filelist = glob2.glob(data_dir + '/**/*_rgb.png')
    print('Total {} images'.format(len(filelist)))

    for curidx in range(len(filelist)):
        if curidx % 100 ==0:
            print('Processing image number', curidx)
        name = filelist[curidx]
        frameindex = name[-12:-8]
        # name = '/home/sicong/detail_data/data/3/0235_rgb.png'
        try:
            img_full = io.imread(name)
        except:
            continue
        depth_full = io.imread(name[0:-8] + '_depth.png')
        depthcount = np.sum(depth_full > 100)
        if depthcount < 100 * 100:
            continue
        rot = 0
        scale = util.getScale_detail(depth_full)
        center = util.getCenter_detail(depth_full)
        if (center[0] < 1 or center[1] < 1 or center[1] > img_full.shape[0] or center[0] > img_full.shape[1]):
            continue

        ori_mask = depth_full > 100
        pcd = PointCloud(np.expand_dims(depth_full, 0), np.expand_dims(ori_mask, 0))
        ori_normal = pcd.get_normal().squeeze(0)
        gt_normal = util_detail.cropfor3d(ori_normal, center, scale, rot, 256, 'nearest')
        gt_normal = normal_util.normalize(gt_normal)
        normal_file_name = name[0:-8] + '_normal.npy'
        np.save(normal_file_name, gt_normal)

if __name__ == '__main__':
    data_dir = '/local-scratch2/normal_dataset/real_data_2'
    produce_normal_file(data_dir)