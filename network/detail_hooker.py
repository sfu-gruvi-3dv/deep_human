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


class DataGenerator():

    def __init__(self, train_dir=None, valid_dir=None, test_dir=None, bg_dir=None, meanRgb_dir=None, show=False,
                 needdepth=False, batch_size=1, sample_set='train', datatype='surreal', depthres=256, seg_joint_res=64):
        """ Initializer
		Args:
			train_dir           : Directory containing training set
			test_dir            : Directory contatining testing set
			valid_dir           : DIrectory contatining validation set

		"""
        self.batch_size = batch_size
        self.sample_set = sample_set
        self.datatype = datatype
        self.depthres = depthres
        self.seg_joint_res = seg_joint_res
        self.joints_num = 16
        self.seg_num = 15
        self.Zres = 59
        self.Zres_joint = 19
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.valid_dir = valid_dir
        self.loadsize = [240, 320]
        self.insize = [256, 256]
        self.outsize = [64, 64]
        self.normalres = [256, 256]
        self.scale = 0.25
        self.rotate = 30
        self.meanRgb_dir = meanRgb_dir
        self.show = show
        self.stp = 0.02
        self.stp_joint = 0.045
        self.halfrange = 0.5
        self.needdepth = needdepth
        self.bg_dir = bg_dir
        self.needlist = True
        self.currentindex = 0
        self.filelist = []
        self.datanum = 0
        # [pelvis, right hip, left hip, right knee, left knee, chest, right foot, left foot, neck, head, right shoulder, left shoulder, right elbow,
        #  left elbow, right hand, left hand]
        self.joints_subset = [0, 1, 2, 4, 5, 6, 7, 8, 12, 15, 16, 17, 18, 19, 22, 23]
        self.joints_subset_h36m = [0, 6, 1, 7, 2, 12, 8, 3, 16, 14, 17, 25, 18, 26, 19, 27]
        self.batches = []

    def _reset_filelist(self):
        sample_set = self.sample_set
        datatype = self.datatype
        print(datatype)
        print(sample_set)
        print(self.train_dir)
        if sample_set == 'train':
            if datatype == 'detail_data':
                self.filelist = glob2.glob(self.train_dir + '/**/*_rgb.npy')
                random.shuffle(self.filelist)
        if sample_set == 'valid':
            if datatype == 'detail_data':
                self.filelist = glob2.glob(self.valid_dir + '/**/*_rgb.npy')
                random.shuffle(self.filelist)
        if sample_set == 'test':
            if datatype == 'detail_data':
                self.filelist = glob2.glob(self.test_dir + '/**/*_rgb.npy')
                random.shuffle(self.filelist)
        self.currentindex = 0
        self.datanum = len(self.filelist)

    def _aux_generator(self, dummyinput=0):
        """ Auxiliary Generator
        Args:
            See Args section in self._generator
        """
        batch_size = self.batch_size
        sample_set = self.sample_set
        datatype = self.datatype
        depthres = self.depthres
        seg_joint_res = self.seg_joint_res

        generated_batch = {}
        random.seed(time.clock())
        self.currentindex = random.randint(0, self.datanum-1)
        generated_batch['train_img'] = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        generated_batch['train_gtdepthre'] = np.zeros((batch_size, depthres, depthres), dtype=np.float32)
        generated_batch['train_mask'] = np.zeros([batch_size, depthres, depthres], dtype=np.bool)

        i = 0
        while i < batch_size:
            if datatype == 'detail_data':
                name = self.filelist[self.currentindex]
                #name = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/detail_data/9/0298_rgb.png'
                header = name[:-8]
                try:
                    generated_batch['train_img'][i] = np.load(name)
                    generated_batch['train_gtdepthre'][i] = np.load(header+'_final_depth.npy')
                    generated_batch['train_mask'][i] = np.load(header + '_mask.npy')
                except:
                    self.currentindex = random.randint(0, self.datanum-1)
                    continue


            # if self.show:
            #     img = Image.fromarray(util.restore_rgb(generated_batch['train_img'],meanRgb_dir), 'RGB')
            #     img.show()


            i = i + 1
            self.currentindex = random.randint(0, self.datanum-1)

        return generated_batch


if __name__ == '__main__':

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'

    train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025'
    valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025'
    test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/data_1025'

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/processed'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/test'
    # valid_dir = '/home/sicong/detail_data/data'
    # test_dir = '/home/sicong/detail_data/data'

    bg_dir = '/local-scratch2/normal_dataset/bg_dataset'

    meanRgb_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/home_bak/surreal/meanRgb.t7'
    #threadsnum = 10

    # generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir,
    #                                         meanRgb_dir, False, False, 1, data_split_type
    #                                         , dataset_type, 256, 64)
    generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir, meanRgb_dir, True, True, datatype='detail_data')
    generator._reset_filelist()
    print('reset done! ', generator.datanum, ' files in dataset')
    batch = generator._aux_generator(10)

    img_batch = batch['train_img']
    img_unnormalize = util.restore_rgb(img_batch,meanRgb_dir)
    img_out = Image.fromarray(img_unnormalize[0],'RGB')
    img_out.show()

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
