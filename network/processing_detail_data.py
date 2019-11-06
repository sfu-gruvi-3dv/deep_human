import numpy as np
import sys

if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import os
import random
import scipy.io as sio
import util, util_detail
from PIL import Image
import matplotlib.pyplot as plt
import time
from torch.utils.serialization import load_lua
from skimage import io
import visualizer
import math
import scipy
import glob2
import scipy.ndimage as ndimage
import multiprocessing as mp
import h5py

def getsubfolders(path):
    list = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            if not file.startswith('.'):
                m = os.path.join(path, file)
                if (os.path.isdir(m)):
                    h = os.path.split(m)
                    list.append(h[1])
    return list

def process_detailfolder(index):
    """ Auxiliary Generator
    Args:
        See Args section in self._generator
    """
    needdepth = True
    insize = [256, 256]
    show = False
    filelist = glob2.glob('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025/' + str(index) + '/**/*_rgb.png')
    datanum = len(filelist)
    print(datanum, ' files in this folder!')
    currentindex = 0
    sample_set = 'train'
    datatype = 'detail_data'
    meanstd = load_lua('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/home_bak/surreal/meanRgb.t7')

    while currentindex < datanum:
        if currentindex % 100 == 0:
            print('processed ', str(currentindex), ' files')
        if datatype == 'detail_data':
            name = filelist[currentindex]
            header = name[:-8]
            try:
                img_full = io.imread(name)
            except:
                currentindex = currentindex + 1
                continue
        if show:
            img = Image.fromarray(img_full, 'RGB')
            img.show()

        depth_full = io.imread(name[0:-8] + '_depth.png')
        depthcount = np.sum(depth_full > 100)
        if depthcount < 100 * 100:
            currentindex = currentindex + 1
            continue

        # crop, scale
        rot = 0
        scale = util.getScale_detail(depth_full)
        center = util.getCenter_detail(depth_full)

        if (center[0] < 1 or center[1] < 1 or center[1] > img_full.shape[0] or center[0] > img_full.shape[1]):
            currentindex = currentindex + 1
            continue

        ## for rgb image
        if datatype == 'detail_data':
            img = util_detail.cropfor3d(img_full, center, scale, rot, insize[1], 'bilinear')
        if img is None:
            currentindex = currentindex + 1
            continue
        if (img.shape[0] == 0 or img.shape[1] == 0):
            currentindex = currentindex + 1
            continue

        if show:
            imgnew = Image.fromarray(img, 'RGB')
            imgnew.show()

        # color augmentation
        img_bak = img
        img = img.astype(np.float32)
        if sample_set == 'train':
            for j in range(3):
                img[:, :, j] = np.clip(img[:, :, j].astype(np.float32) / 255 * np.random.uniform(0.6, 1.4), 0.0,
                                       1.0)
        else:
            for j in range(3):
                img[:, :, j] = np.clip(img[:, :, j].astype(np.float32) / 255, 0.0, 1.0)
        # print('color augmentation done!')

        # whitening rgb image
        for j in range(3):
            img[:, :, j] = img[:, :, j] - meanstd['mean'][j]
            img[:, :, j] = img[:, :, j] / meanstd['std'][j]

        np.save(header+'_rgb.npy', img)

        ## for depth
        if datatype == 'detail_data' and needdepth:
            depm_continue = util_detail.cropfor3d(depth_full, center, scale, rot, insize[1], 'bilinear')

            if show:
                plt.figure()
                plt.imshow(depth_full, aspect='auto', cmap=plt.get_cmap('jet'))
                plt.show()

            if show:
                plt.figure()
                plt.imshow(depm_continue, aspect='auto', cmap=plt.get_cmap('jet'))
                plt.show()
            mask = depm_continue > 100
            depm_continue[depm_continue < 100] = 15 * 1000.0
            final_depth = depm_continue / 1000.0
            median_value = np.median(final_depth[final_depth < 5])
            final_depth = final_depth - median_value + 0.10
            final_depth[final_depth > 5] = 0.60
            np.save(header + '_final_depth.npy', final_depth)
            np.save(header + '_mask.npy', mask)

            if show:
                plt.figure()
                plt.imshow(mask, aspect='auto', cmap=plt.get_cmap('jet'))
                plt.show()

        if show and needdepth:
            plt.figure()
            plt.imshow(final_depth, aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()

        ## for 2d segmentation
        currentindex = currentindex + 1

    return


if __name__ == '__main__':
    # rgbfilelist = glob2.glob('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025/' + '/**/*_rgb.png')
    # print(len(rgbfilelist), ' rgb files')
    #
    # rgbnpyfilelist = glob2.glob('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025/' + '/**/*_rgb.npy')
    # print(len(rgbnpyfilelist), ' rgb npy files')
    #
    # depthfilelist = glob2.glob('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025/' + '/**/*_final_depth.npy')
    # print(len(depthfilelist), ' depth files')
    #
    # maskfilelist = glob2.glob('/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025/' + '/**/*_mask.npy')
    # print(len(maskfilelist), ' mask files')




    # threadnum = 24
    # pool = mp.Pool(threadnum)
    # pool.map(process_detailfolder, range(threadnum))
    # # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    #
    # # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    #
    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/detail_data'
    # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/detail_data'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/detail_data'
    #
    # # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/processed'
    # # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/test'
    # # valid_dir = '/home/sicong/detail_data/data'
    # # test_dir = '/home/sicong/detail_data/data'
    #
    # bg_dir = '/local-scratch2/normal_dataset/bg_dataset'
    #
    # meanRgb_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/home_bak/surreal/meanRgb.t7'
    # #threadsnum = 10
    #
    # # generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir,
    # #                                         meanRgb_dir, False, False, 1, data_split_type
    # #                                         , dataset_type, 256, 64)
    # generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir, meanRgb_dir, True, True, datatype='detail_data')
    # generator._reset_filelist()
    # print('reset done! ', generator.datanum, ' files in dataset')
    # batch = generator._aux_generator(10)
    # img_batch = batch['train_img']
    # depth_batch = batch['train_gtdepthre'][0]
    # mask = batch['train_mask'][0]
    #
    # meanstd = load_lua(meanRgb_dir)
    #
    # img = img_batch[0]
    # for j in range(3):
    #     img[:, :, j] = img[:, :, j] * meanstd['std'][j]
    #     img[:, :, j] = img[:, :, j] + meanstd['mean'][j]
    #
    #
    # f = open("depth_" + ".ply", "w")
    # f.write("ply\n")
    # f.write("format ascii 1.0\n")
    # f.write("element vertex " + str(256 * 256) + "\n")
    # f.write("property float x\n")
    # f.write("property float y\n")
    # f.write("property float z\n")
    # f.write("property uchar red\n")
    # f.write("property uchar green\n")
    # f.write("property uchar blue\n")
    # f.write("end_header\n")
    #
    # # img2 = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_NEAREST)
    # for i in range(256):
    #     for j in range(256):
    #         if mask[j, i] > 0:
    #             z = depth_batch[j, i]
    #         else:
    #             z = 0.6
    #         f.write(str((i - 128) / 128) + "  " + str((j - 128) / 128) + "  " + str(z) + " " +
    #                 str(int(img[j, i, 0] * 255)) + " " +
    #                 str(int(img[j, i, 1] * 255)) + " " +
    #                 str(int(img[j, i, 2] * 255)) + "\n")
    #
    # f.close()

    # meanstd = load_lua(generator.meanRgb_dir)
    # img_show = img_batch[0]
    # for j in range(3):
    #     img_show[:, :, j] = img_show[:, :, j] * meanstd['std'][j]
    #     img_show[:, :, j] = img_show[:, :, j] + meanstd['mean'][j]
    # img_show = img_show * 255.0
    # img_show = img_show.astype(np.uint8)
    # img_out = Image.fromarray(img_show)
    # img_out.show()

    # allbatches = []
    # pool = mp.Pool(threadsnum)
    # for i in range(5):
    #     batches = pool.map(generator._aux_generator, range(threadsnum))
    #     allbatches.extend(batches)
    #     print(len(allbatches), " batches in the list")
