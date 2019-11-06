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
                 needdepth=False, batch_size=1, sample_set='train', datatype='surreal', depthres=256, seg_joint_res=64, readnpy = False):
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
        self.readnpy = readnpy
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
            if datatype == 'up-3d':
                self.filelist = glob2.glob(self.train_dir + '/pose_prepared/91/500/up-p91/' + '/**/*_image.png')
                random.shuffle(self.filelist)
            elif datatype == 'normal_dataset':
                self.filelist = glob2.glob(self.train_dir + '/**/*.tiff')
                random.shuffle(self.filelist)
            elif datatype == 'detail_data':
                self.filelist = glob2.glob(self.train_dir + '/**/*_rgb.png')
                random.shuffle(self.filelist)
            elif datatype == 'realtest':
                self.filelist = glob2.glob(self.train_dir + '/**/*.jpg')
                random.shuffle(self.filelist)
            elif datatype == 'human3.6m':
                self.filelist = glob2.glob(self.train_dir + '/**/*.h5')
                random.shuffle(self.filelist)
            else:
                self.filelist = glob2.glob(self.train_dir + '/**/*.mp4')
                random.shuffle(self.filelist)
        if sample_set == 'valid':
            if datatype == 'up-3d':
                self.filelist = glob2.glob(self.valid_dir + '/pose_prepared/91/500/up-p91/' + '/**/*_image.png')
                random.shuffle(self.filelist)
            elif datatype == 'normal_dataset':
                self.filelist = glob2.glob(self.valid_dir + '/**/*.tiff')
                random.shuffle(self.filelist)
            elif datatype == 'detail_data':
                self.filelist = glob2.glob(self.valid_dir + '/**/*_rgb.png')
                random.shuffle(self.filelist)
            elif datatype == 'realtest':
                self.filelist = glob2.glob(self.valid_dir + '/**/*.jpg')
                random.shuffle(self.filelist)
            elif datatype == 'human3.6m':
                self.filelist = glob2.glob(self.valid_dir + '/**/*.h5')
                random.shuffle(self.filelist)
            else:
                self.filelist = glob2.glob(self.valid_dir + '/**/*.mp4')
                random.shuffle(self.filelist)
        if sample_set == 'test':
            if datatype == 'up-3d':
                self.filelist = glob2.glob(self.test_dir + '/pose_prepared/91/500/up-p91/' + '/**/*_image.png')
                random.shuffle(self.filelist)
            elif datatype == 'normal_dataset':
                self.filelist = glob2.glob(self.test_dir + '/**/**.tiff')
                random.shuffle(self.filelist)
            elif datatype == 'detail_data':
                self.filelist = glob2.glob(self.test_dir + '/**/*_rgb.png')
                # random.shuffle(self.filelist)
            elif datatype == 'realtest':
                self.filelist = glob2.glob(self.test_dir + '/**/*.jpg')
                random.shuffle(self.filelist)
            elif datatype == 'human3.6m':
                self.filelist = glob2.glob(self.test_dir + '/**/*.h5')
                random.shuffle(self.filelist)
            else:
                self.filelist = glob2.glob(self.test_dir + '/**/*.mp4')
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
        generated_batch['train_img'] = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
        generated_batch['train_gtseg'] = np.zeros([batch_size, seg_joint_res, seg_joint_res], dtype=np.int8)
        generated_batch['train_gt2dheat'] = np.zeros([batch_size, seg_joint_res, seg_joint_res, self.joints_num],
                                                     dtype=np.float32)
        generated_batch['train_gtjoints'] = np.zeros((batch_size, 64, 64, self.joints_num * self.Zres_joint),
                                                     dtype=np.float32)
        generated_batch['train_gtdepthre'] = np.zeros((batch_size, depthres, depthres), dtype=np.float32)
        generated_batch['train_mask'] = np.zeros([batch_size, depthres, depthres], dtype=np.bool)
        #generated_batch['train_2djoints'] = np.zeros([batch_size, 2, self.joints_num], dtype=np.float32)
        generated_batch['train_3djoints'] = np.zeros([batch_size, 3, self.joints_num], dtype=np.float32)

        # For evaluation
        generated_batch['joints_4_eval'] = np.zeros([batch_size,3,self.joints_num],dtype=np.float32)
        generated_batch['intrinsic'] = np.zeros([batch_size,3,3],dtype=np.float32)
        generated_batch['scale'] = np.zeros([batch_size],dtype=np.float32)
        generated_batch['center'] = np.zeros([batch_size,2], dtype=np.float32)
        generated_batch['zroot'] = np.zeros([batch_size],dtype=np.float32)
        generated_batch['cam_loc'] = np.zeros([batch_size,3,1],dtype=np.float32)

        i = 0
        if datatype == 'normal_dataset':
            generated_batch['normal_train_img'] = np.zeros((batch_size, self.normalres[0], self.normalres[1], 3),
                                                           dtype=np.float32)
            generated_batch['normal_train_gtnormal'] = np.zeros([batch_size, self.normalres[0], self.normalres[1], 3],
                                                                dtype=np.float32)
            generated_batch['normal_train_gtdepthre'] = np.zeros((batch_size, self.normalres[0], self.normalres[1]),
                                                                 dtype=np.float32)
            generated_batch['normal_train_mask'] = np.zeros([batch_size, self.normalres[0], self.normalres[1]],
                                                            dtype=np.bool)
            while i < batch_size:
                img_name = self.filelist[self.currentindex]
                type_dir = os.path.join(self.test_dir,
                                        img_name.split('/')[-4])  # random.sample(getsubfolders(self.train_dir), 1)[0])
                depth_dir = type_dir + '/depth_maps'
                normal_dir = type_dir + '/normals'

                view_type = img_name.split('/')[-2]

                depth_dir = os.path.join(depth_dir, view_type)
                normal_dir = os.path.join(normal_dir, view_type)

                index = img_name[-9:-5]
                depth_name = depth_dir + '/depth_' + index + '.npz'
                normal_name = normal_dir + '/normals_' + index + '.npz'

                bg_name = os.path.join(self.bg_dir, random.sample(os.listdir(self.bg_dir), 1)[0])
                bg_name = os.path.join(bg_name, random.sample(os.listdir(bg_name), 1)[0])

                try:
                    bg_img = io.imread(bg_name)
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
                bg_img = scipy.misc.imresize(bg_img, [self.normalres[0], self.normalres[1]], interp='bilinear')
                img = io.imread(img_name)
                nmap = np.load(normal_name)['normals']
                dmap = np.load(depth_name)['depth']
                mask = dmap > 1e-4

                generated_mask = np.zeros([self.normalres[0], self.normalres[1]], dtype=np.bool)
                generated_mask[15:239, 15:239] = mask
                generated_batch['normal_train_mask'][i] = generated_mask
                img_pad = np.zeros((self.normalres[0], self.normalres[1], 3), dtype=np.uint8)
                img_pad[15: 239, 15: 239, :] = img.astype(np.float32)
                bg_img[generated_mask] = img_pad[generated_mask]

                # plt.figure()
                # plt.imshow(bg_img, aspect='auto',
                #            cmap=plt.get_cmap('jet'))
                # plt.show()

                bg_img = bg_img.astype(np.float32)
                # color augmentation
                if sample_set == 'train':
                    for j in range(3):
                        bg_img[:, :, j] = np.clip(
                            bg_img[:, :, j].astype(np.float32) / 255 * np.random.uniform(0.6, 1.4), 0.0,
                            1.0)
                else:
                    for j in range(3):
                        bg_img[:, :, j] = np.clip(bg_img[:, :, j].astype(np.float32) / 255, 0.0, 1.0)
                # print('color augmentation done!')

                # whitening rgb image
                meanstd = load_lua(self.meanRgb_dir)
                for j in range(3):
                    bg_img[:, :, j] = bg_img[:, :, j] - meanstd['mean'][j]
                    bg_img[:, :, j] = bg_img[:, :, j] / meanstd['std'][j]
                generated_batch['normal_train_img'][i, :, :, :] = bg_img

                generated_batch['normal_train_gtnormal'][i, 15:239, 15:239, :] = nmap

                if self.show:
                    plt.figure()
                    plt.imshow(generated_batch['normal_train_gtnormal'][i, :, :, 0], aspect='auto',cmap=plt.get_cmap('jet'))
                    plt.show()

                generated_batch['normal_train_gtdepthre'][i, 15:239, 15:239] = dmap

                i = i + 1

                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue
            return generated_batch

        if datatype == 'human3.6m':
            camdic = {'54138969': 0, '55011271': 1, '58860488': 2, '60457274': 3}
            while i < batch_size:
                #print('len(self.filelist): ',len(self.filelist),'self.currentindex: ',self.currentindex, 'self.datanum: ',self.datanum)
                name = self.filelist[self.currentindex]
                #name = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/h36m-fetch/processed/S9/Greeting-2/annot.h5'
                h5file = h5py.File(name,'r')
                j3dbuff = h5file['pose']['3d']
                j2dbuff = h5file['pose']['2d']

                framenum = h5file['frame'].shape[0]
                idx = random.randint(0,framenum-1)
                #idx = 57
                joints2dfull = j2dbuff[idx,:,:]
                joints3dfull = j3dbuff[idx,:,:]

                joints2d = joints2dfull[self.joints_subset_h36m,:].transpose()
                joints3d = joints3dfull[self.joints_subset_h36m,:].transpose() / 1000.0
                generated_batch['joints_4_eval'][i] = joints3d

                imgbank_dir = name[:-8] + 'imageSequence/'
                foldernum = len(os.listdir(imgbank_dir))
                imgbank_dir = imgbank_dir + os.listdir(imgbank_dir)[idx // (framenum // foldernum)]
                imgname = imgbank_dir + '/img_%06d.jpg' % h5file['frame'][idx]
                img_full = io.imread(imgname) ##io.imread(imgname)
                indexbuff = imgname.split('/')

                frameindex = h5file['frame'][idx]
                action = indexbuff[-4]
                subaction = indexbuff[-2]
                find_intrisic = False
                for k, subi in list(h5file['intrinsics'].items()):
                    if subaction == k:
                        intrinsic = subi.value
                        find_intrisic = True
                        break
                if find_intrisic == False:
                    print('something wrong here!')
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
                generated_batch['intrinsic'][i, 0, 0] = intrinsic[0]
                generated_batch['intrinsic'][i, 0, 2] = intrinsic[1]
                generated_batch['intrinsic'][i, 1, 1] = intrinsic[2]
                generated_batch['intrinsic'][i, 1, 2] = intrinsic[3]
                fold = indexbuff[-5]

                if action.split('-')[1] == '2':
                    action = action.split('-')[0]
                else:
                    action = action.split('-')[0] + ' 1'
                if self.datatype == 'train':
                    segname = self.train_dir +'/segmentation/' + fold + '/MySegmentsMat/PartLabels/'+ action + '.' + subaction + '.mat'
                else:
                    segname = self.test_dir +'/segmentation/' + fold + '/MySegmentsMat/PartLabels/'+ action + '.' + subaction + '.mat'
                try:
                    segs = h5py.File(segname, 'r')
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
                try:
                    segm_full = np.transpose(segs[segs['Feat'][frameindex][0]])
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue


                if self.show:
                    plt.figure()
                    plt.imshow(segm_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                    plt.figure()
                    plt.imshow(img_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                    #visualizer.show_prections(img_full,joints2d.transpose())




                rot = 0
                scale = util.getScale(joints2d)
                center = util.getCenter(joints2d)
                generated_batch['center'][i] = center
                generated_batch['scale'][i] = scale
                if (center[0] < 1 or center[1] < 1 or center[1] > img_full.shape[0] or center[0] > img_full.shape[1]):
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue

                img = util.cropfor3d(img_full, center, scale, rot, self.insize[1], 'bilinear')

                if img is None:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
                if (img.shape[0] == 0 or img.shape[1] == 0):
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue

## for 2d heatmap
                sigma_2d_inscale = math.floor(2 * self.insize[0] / self.outsize[0])
                out_2d = np.zeros([self.insize[0], self.insize[1], self.joints_num])

                for j in range(self.joints_num):
                    pt = util.transform(joints2d[:, j], center, scale, 0, self.insize[0], False)
                    generated_batch['train_3djoints'][i, 0:2, j] = np.asarray([pt[1],pt[0]]) / (self.insize[0] / seg_joint_res)
                    heat_slice = util.Drawgaussian2D(img, pt, sigma_2d_inscale)
                    # if np.sum(heat_slice) > 1e-2:
                    #     heat_slice /= np.sum(heat_slice)
                    # else:
                    #     heat_slice *= 0
                    # print('heat_slice.shape',heat_slice.shape)

                    out_2d[:, :, j] = heat_slice
                    # if self.show:
                    #     plt.figure()
                    #     plt.imshow(heat_slice, aspect='auto', cmap=plt.get_cmap('jet'))
                    #     plt.show()

                out_2d = cv2.resize(out_2d, (seg_joint_res, seg_joint_res), interpolation=cv2.INTER_NEAREST)
                generated_batch['train_gt2dheat'][i] = out_2d
                if self.show:
                    visualizer.draw2dskeleton(img.astype(np.uint8), out_2d)
                    visualizer.draw2dskeleton_joints(img.astype(np.uint8), generated_batch['train_3djoints'][i, 0:2, :] )
## for 3d heat map

                dPelvis = joints3d[2, 0]
                quantized_joints3d, ctn_joints3d = util.quantize(joints3d[2, :], dPelvis, self.stp_joint, self.Zres_joint)
                generated_batch['train_3djoints'][i, 2, :] = ctn_joints3d
                generated_batch['zroot'][i]=dPelvis
                out = np.zeros([self.outsize[0], self.outsize[1], self.joints_num * self.Zres_joint])
                sigma_2d = 2
                size_z = 2 * math.floor((6 * sigma_2d * self.Zres_joint / self.outsize[0] + 1) / 2) + 1
                for j in range(self.joints_num):
                    # if joints2d[1,j] >= img_full.shape[0] or joints2d[0,j] >=img_full.shape[1] or joints2d[1,j]<0 or joints2d[0,j]<0:
                    # continue
                    z = quantized_joints3d[j]
                    pt = util.transform(joints2d[:, j], center, scale, 0, self.outsize[0], False)
                    out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint] = util.Drawguassian3D(
                        out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint], pt, z, sigma_2d, size_z)
                    # if self.show:
                    #     util.save3dheat(out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint], j)

                    generated_batch['train_gtjoints'][i] = out

                if self.show:
                    visualizer.draw3dskeleton(self.joints_num, self.Zres_joint, out)
                    visualizer.draw3dskeleton_joints(16, 19, generated_batch['train_3djoints'][i])
## for segmentation
                # for j in range(80):
                #     one_hot_seg = np.zeros(segm_full.shape)
                #     one_hot_seg[segm_full == j] = 1
                #     onehot_segm = util.cropfor3d(one_hot_seg, center, scale, rot, self.insize[1], 'nearest')
                #     segm = util.cropfor3d(segm_full, center, scale, rot, self.insize[1], 'nearest')
                #     segm = cv2.resize(segm, (seg_joint_res, seg_joint_res),
                #                                                          interpolation=cv2.INTER_NEAREST)
                #
                #
                #     if self.show:
                #         print('show part: ',j)
                #         plt.figure()
                #         plt.imshow(segm, aspect='auto', cmap=plt.get_cmap('jet'))
                #         plt.show()
                #
                #         plt.figure()
                #         plt.imshow(onehot_segm, aspect='auto', cmap=plt.get_cmap('jet'))
                #         plt.show()


                # one_hot_seg = np.zeros(segm_full.shape)
                # one_hot_seg[segm_full == 38] = 1


                if self.show:
                    plt.figure()
                    plt.imshow(segm_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                # if self.show:
                #     plt.figure()
                #     plt.imshow(one_hot_seg, aspect='auto', cmap=plt.get_cmap('jet'))
                #     plt.show()


                segm_full = util.changeSegmIxup3d(segm_full,
                                              [
                                                  0,2,12,14,0,0,2,9,11,0,
                                                  0,0,0,0,0,0,0,3,4,5,
                                                  0,0,0,0,0,6,7,8,0,0,
                                                  0,0,2,12,13,0,0,0,9,10,
                                                  0,0,0,2,2,0,1,0,0,3,
                                                  4,0,0,0,0,0,6,0,7,0
                                              ]).astype(np.int8)

                segm = util.cropfor3d(segm_full, center, scale, rot, self.insize[1], 'nearest')


                generated_batch['train_gtseg'][i, :, :] = cv2.resize(segm, (seg_joint_res, seg_joint_res),
                                                                     interpolation=cv2.INTER_NEAREST)

                if self.show:
                    plt.figure()
                    plt.imshow(segm, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                if self.show:
                    imgnew = Image.fromarray(img, 'RGB')
                    imgnew.show()

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
                meanstd = load_lua(self.meanRgb_dir)
                for j in range(3):
                    img[:, :, j] = img[:, :, j] - meanstd['mean'][j]
                    img[:, :, j] = img[:, :, j] / meanstd['std'][j]



                generated_batch['train_img'][i] = cv2.resize(img.astype(np.float32), (256, 256), interpolation=cv2.INTER_NEAREST)

                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue
                i = i+1
            return generated_batch

        if datatype == 'realtest':
            while i < batch_size:
                name = self.filelist[self.currentindex]
                testimg = io.imread(name)
                testimg = scipy.misc.imresize(testimg, [self.insize[1], self.insize[1]], interp='bilinear').astype(
                    np.float32)
                meanstd = load_lua(self.meanRgb_dir)
                for j in range(3):
                    testimg[:, :, j] = np.clip(testimg[:, :, j].astype(np.float32) / 255.0, 0.0, 1.0)
                    testimg[:, :, j] = testimg[:, :, j] - meanstd['mean'][j]
                    testimg[:, :, j] = testimg[:, :, j] / meanstd['std'][j]
                generated_batch['train_img'][i] = cv2.resize(testimg, (self.insize[0], self.insize[1]),
                                                             interpolation=cv2.INTER_NEAREST)
                i += 1
                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue

                if self.show:
                    plt.figure()
                    plt.imshow(generated_batch['train_img'][0], aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

            return generated_batch

        while i < batch_size:
            if datatype != 'detail_data' and datatype != 'up-3d':
                name = self.filelist[self.currentindex]
                #print(name)
                #name = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/test/run1/ung_120_03/ung_120_03_c0008.mp4'
                cap = cv2.VideoCapture(name)
                length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frameindex = random.randint(1, length)
                #print(frameindex)
                #frameindex = 21
                cap.set(1, frameindex - 1)
                _, img_full = cap.read()
                try:
                    img_full = cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB)
                except:
                    continue
                bodyinfo = sio.loadmat(name[0:-4] + '_info.mat')

            elif datatype == 'detail_data':
                name = self.filelist[self.currentindex]
                print(name)
                #name = '/home/sicong/forfusion/testdataset/9/0055_rgb.png'
                frameindex = name[-12:-8]
                try:
                    img_full = io.imread(name)[:,:,0:3]
                    # print('img_full.shape',img_full.shape)
                    # exit()
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue

            elif datatype == 'up-3d':
                if sample_set == 'train':
                    info_dir = self.train_dir + '/pose_prepared/91/500/up-p91/'
                    seg_dir = self.train_dir + '/segment/up-s31/s31/'
                elif sample_set == 'valid':
                    info_dir = self.valid_dir + '/pose_prepared/91/500/up-p91/'
                    seg_dir = self.valid_dir + '/segment/up-s31/s31/'
                elif sample_set == 'test':
                    info_dir = self.test_dir + '/pose_prepared/91/500/up-p91/'
                    seg_dir = self.test_dir + '/segment/up-s31/s31/'

                name = self.filelist[self.currentindex]
                # name = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets/pose_prepared/91/500/up-p91/04877_image.png'
                frameindex = name[-15:-10]
                try:
                    img_full = io.imread(name)
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
                try:
                    bodyinfo = sio.loadmat(info_dir + frameindex + '_info.mat')
                except:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue
            if self.show:
                img = Image.fromarray(img_full, 'RGB')
                img.show()

            if datatype != 'detail_data':
                # load 2d joints to determine the bounding box
                # [2 x njoints]
                if datatype != 'up-3d':
                    if bodyinfo is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints2dfull = bodyinfo['joints2D']
                    if joints2dfull is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    if len(joints2dfull.shape) < 3:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    if frameindex - 1 >= joints2dfull.shape[2]:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints2d = joints2dfull[:, self.joints_subset, frameindex - 1].astype(np.int64)

                    joints3dfull = bodyinfo['joints3D']
                    if joints3dfull is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    if frameindex - 1 >= joints2dfull.shape[2]:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints3d = joints3dfull[:, self.joints_subset, frameindex - 1]
                    # generated_batch['train_2djoints'][i, :] = joints2d
                    # generated_batch['train_3djoints'][i, :] = joints3d

                    depth_full = sio.loadmat(name[0:-4] + '_depth.mat')['depth_' + str(frameindex)]
                elif datatype == 'up-3d':
                    if bodyinfo is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints2dfull = bodyinfo['joints2D']
                    if joints2dfull is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    if len(joints2dfull.shape) < 2:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints2d = joints2dfull[:, self.joints_subset].astype(np.int64)
                    joints3dfull = np.transpose(bodyinfo['joints3D'])
                    if joints3dfull is None:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                    joints3d = joints3dfull[:, self.joints_subset]

                    depth_full = sio.loadmat(info_dir + frameindex + '_depth.mat')['depth']

                # set pelvis as the original point
                generated_batch['joints_4_eval'][i] = joints3d
                #generated_batch['cam_loc'][i] = bodyinfo['camLoc']
                camLoc = bodyinfo['camLoc'][0]
                if datatype == 'up-3d':
                    dPelvis = joints3d[2, 0]
                else:
                    camlocation = camLoc
                    joints3d[0, :] = camlocation - joints3d[0, :]
                    dPelvis = joints3d[0, 0]

                if datatype != 'up-3d':
                    segm_raw = sio.loadmat(name[0:-4] + '_segm.mat')['segm_' + str(frameindex)]

                    segm_full = util.changeSegmIx(segm_raw,
                                                  [2, 12, 9, 2, 13, 10, 2, 14, 11, 2, 14, 11, 2, 2, 2, 1, 6, 3, 7, 4, 8,
                                                   5, 8,
                                                   5]).astype(np.int8)

                else:
                    segm_raw = cv2.imread(seg_dir + frameindex + '_ann_vis.png')
                    segm_full = util.up3dtosurreal(segm_raw)

                if self.show:
                    plt.figure()
                    plt.imshow(segm_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                if datatype == 'up-3d':
                    quantized_joints3d, ctn_joints3d = util.quantize(joints3d[2, :], dPelvis, self.stp_joint, self.Zres_joint)
                    generated_batch['train_3djoints'][i, 2, :] = ctn_joints3d
                    generated_batch['zroot'][i] = dPelvis
                    if self.needdepth:
                        relative_depth, _ = util.relative_up3d(depth_full, dPelvis, self.stp,
                                                               self.Zres)  # self.halfrange
                elif datatype != 'detail_data':
                    quantized_joints3d, ctn_joints3d = util.quantize(joints3d[0, :], dPelvis, self.stp_joint, self.Zres_joint)
                    generated_batch['train_3djoints'][i, 2, :] = ctn_joints3d
                    generated_batch['zroot'][i] = dPelvis
                    if self.needdepth:
                        relative_depth, _ = util.relative(depth_full, dPelvis, self.stp, self.Zres)  # self.halfrange

                # TODO: 1. resize quantized_depth 2. output dense continuous relative depth in util.quantize
                if self.show and self.needdepth:
                    plt.figure()
                    plt.imshow(depth_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()
                if self.show and self.needdepth:
                    plt.figure()
                    plt.imshow(relative_depth, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()
            else:
                if self.readnpy == False:
                    depth_full = io.imread(name[0:-8] + '_depth.png')
                    # depth_full_smooth = util.Kinect_DepthNormalization(depth_full)
                    # depth_full_smooth[depth_full < 100] = 0
                    # depth_full = depth_full_smooth
                    depthcount = np.sum(depth_full > 100)
                    if depthcount < 100 * 100:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue
                else :
                    depth_full = np.load(name[0:-8] + '_depth.npy')
                    depth_full = depth_full * 1000.0
                    depthcount = np.sum(depth_full > 100)
                    if depthcount < 100 * 100:
                        self.currentindex = self.currentindex + 1
                        if self.currentindex >= self.datanum:
                            random.shuffle(self.filelist)
                            self.currentindex = 0
                        continue



            if datatype != 'detail_data':
                # crop, scale
                rot = 0
                scale = util.getScale(joints2d)
                center = util.getCenter(joints2d)
                generated_batch['scale'][i] = scale
                generated_batch['center'][i] = center
            else:
                # crop, scale
                rot = 0
                scale = util.getScale_detail(depth_full)
                center = util.getCenter_detail(depth_full)

            if (center[0] < 1 or center[1] < 1 or center[1] > img_full.shape[0] or center[0] > img_full.shape[1]):
                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue

            ## for rgb image
            if datatype != 'up-3d' and datatype != 'detail_data':
                img = util.cropfor3d(img_full, center, scale, rot, self.insize[1], 'bilinear')
            elif datatype == 'detail_data':
                img = util_detail.cropfor3d(img_full, center, scale, rot, self.insize[1], 'bilinear')
            elif datatype == 'up-3d':
                norm_factor = np.array([self.insize[1] / img_full.shape[1], self.insize[1] / img_full.shape[0]],
                                       dtype=np.float32)
                img = scipy.misc.imresize(img_full, [self.insize[1], self.insize[1]], interp='bilinear')
                badexample = False
                for j in range(joints2d.shape[1]):
                    joints2d_rescaled = np.multiply(joints2d[:, j], norm_factor).astype(np.int64)
                    if joints2d_rescaled[0] < 0 or joints2d_rescaled[0] > 256 or joints2d_rescaled[1] < 0 or \
                            joints2d_rescaled[1] > 256:
                        badexample = True
                if badexample:
                    self.currentindex = self.currentindex + 1
                    if self.currentindex >= self.datanum:
                        random.shuffle(self.filelist)
                        self.currentindex = 0
                    continue

            if img is None:
                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue
            if (img.shape[0] == 0 or img.shape[1] == 0):
                self.currentindex = self.currentindex + 1
                if self.currentindex >= self.datanum:
                    random.shuffle(self.filelist)
                    self.currentindex = 0
                continue

            if self.show:
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
            meanstd = load_lua(self.meanRgb_dir)
            for j in range(3):
                img[:, :, j] = img[:, :, j] - meanstd['mean'][j]
                img[:, :, j] = img[:, :, j] / meanstd['std'][j]

            generated_batch['train_img'][i] = img

            ## for depth
            if datatype == 'detail_data' and self.needdepth:
                depm_continue = util_detail.cropfor3d(depth_full, center, scale, rot, self.insize[1], 'bilinear')

                if self.show:
                    plt.figure()
                    plt.imshow(depth_full, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                if self.show:
                    plt.figure()
                    plt.imshow(depm_continue, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()
                mask = depm_continue > 100
                depm_continue[depm_continue < 100] = 15 * 1000.0
                final_depth = depm_continue / 1000.0
                median_value = np.median(final_depth[final_depth < 5])
                final_depth = final_depth - median_value + 0.10
                final_depth[final_depth > 5] = 0.60
                generated_batch['train_gtdepthre'][i, :, :] = final_depth

                mask = ndimage.binary_erosion(mask).astype(mask.dtype)
                generated_batch['train_mask'][i, :, :] = mask

                if self.show:
                    plt.figure()
                    plt.imshow(mask, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

            elif datatype == 'up-3d' and self.needdepth:
                depm_continue = cv2.resize(relative_depth.astype(np.float32), (depthres, depthres),
                                           interpolation=cv2.INTER_NEAREST)
                generated_batch['train_gtdepthre'][i, :, :] = depm_continue
                mask = depm_continue < 0.59

                mask = ndimage.binary_erosion(mask).astype(mask.dtype)
                generated_batch['train_mask'][i, :, :] = mask

            elif self.needdepth:
                depm_continue = util.cropdepth(relative_depth, center, scale, rot, self.insize[1], 0.60)
                generated_batch['train_gtdepthre'][i, :, :] = cv2.resize(depm_continue, (depthres, depthres),
                                                                         interpolation=cv2.INTER_NEAREST)
                mask = depm_continue < 0.59

                mask = ndimage.binary_erosion(mask).astype(mask.dtype)
                generated_batch['train_mask'][i, :, :] = mask

            if self.show and self.needdepth:
                plt.figure()
                plt.imshow(generated_batch['train_gtdepthre'][i, :, :], aspect='auto', cmap=plt.get_cmap('jet'))
                plt.show()

            ## for 2d segmentation

            if datatype == 'up-3d':
                segm = cv2.resize(segm_full, (seg_joint_res, seg_joint_res),
                                  interpolation=cv2.INTER_NEAREST)
                generated_batch['train_gtseg'][i, :, :] = segm

            elif datatype != 'detail_data':
                segm = util.cropfor3d(segm_full, center, scale, rot, self.insize[1], 'nearest')
                generated_batch['train_gtseg'][i, :, :] = cv2.resize(segm, (seg_joint_res, seg_joint_res),
                                                                     interpolation=cv2.INTER_NEAREST)
                generated_batch['intrinsic'][i] = np.asarray([[600, 0, 160], [0, 600, 120], [0, 0, 1]],
                                                             dtype=np.float32)
                if self.show:
                    plt.figure()
                    plt.imshow(segm, aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

            # for j in range(16):
            #     one_hot_seg = np.zeros(segm.shape)
            #     one_hot_seg[segm == j] = 1
            #     if self.show:
            #         print('show part: ', j)
            #         plt.figure()
            #         plt.imshow(segm, aspect='auto', cmap=plt.get_cmap('jet'))
            #         plt.show()
            #
            #         plt.figure()
            #         plt.imshow(one_hot_seg, aspect='auto', cmap=plt.get_cmap('jet'))
            #         plt.show()

            ## for 2d joints

            if datatype != 'detail_data':
                # TODO: create 2d heatmaps
                sigma_2d_inscale = math.floor(2 * self.insize[0] / self.outsize[0])
                out_2d = np.zeros([self.insize[0], self.insize[1], self.joints_num])

                for j in range(self.joints_num):
                    if datatype == 'up-3d':
                        pt = np.multiply(joints2d[:, j], norm_factor).astype(np.int64)
                    else:
                        pt = util.transform(joints2d[:, j], center, scale, 0, self.insize[0], False)
                    generated_batch['train_3djoints'][i, 0:2, j] = np.asarray([pt[1],pt[0]],dtype=np.float32)/(self.insize[0]/seg_joint_res)
                    # for debug
                    heat_slice = util.Drawgaussian2D(img, pt, sigma_2d_inscale)
                    # if np.sum(heat_slice) > 1e-2:
                    #     heat_slice /= np.sum(heat_slice)
                    # else:
                    #     heat_slice *= 0
                    # print('heat_slice.shape',heat_slice.shape)

                    out_2d[:, :, j] = heat_slice
                    # if self.show:
                    #     plt.figure()
                    #     plt.imshow(heat_slice, aspect='auto', cmap=plt.get_cmap('jet'))
                    #     plt.show()
                #generated_batch['train_3djoints'][i, 0:2, :] = joints2d
                # print('before cropping:\n', joints2d)
                # print('center: ', center, 'length: ', scale * 200)
                # print('after cropping:\n', generated_batch['train_3djoints'][i, 0:2, :])

                out_2d = cv2.resize(out_2d, (seg_joint_res, seg_joint_res), interpolation=cv2.INTER_NEAREST)
                generated_batch['train_gt2dheat'][i] = out_2d
                if self.show:
                    visualizer.draw2dskeleton(img_bak.astype(np.uint8), out_2d)
                    #print('-----------------------------------------------------------------')
                    visualizer.draw2dskeleton_joints(img_bak.astype(np.uint8),generated_batch['train_3djoints'][i, 0:2, :])


            # for 3d joints
            if datatype != 'detail_data':
                out = np.zeros([self.outsize[0], self.outsize[1], self.joints_num * self.Zres_joint])
                sigma_2d = 2
                size_z = 2 * math.floor((6 * sigma_2d * self.Zres_joint / self.outsize[0] + 1) / 2) + 1
                for j in range(self.joints_num):
                    # if joints2d[1,j] >= img_full.shape[0] or joints2d[0,j] >=img_full.shape[1] or joints2d[1,j]<0 or joints2d[0,j]<0:
                    # continue
                    z = quantized_joints3d[j]
                    if datatype == 'up-3d':
                        pt = np.multiply(joints2d[:, j], norm_factor / 4).astype(np.int64)
                    else:
                        pt = util.transform(joints2d[:, j], center, scale, 0, self.outsize[0], False)
                    out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint] = util.Drawguassian3D(
                        out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint], pt, z, sigma_2d, size_z)
                    if self.show:
                        util.save3dheat(out[:, :, j * self.Zres_joint: (j + 1) * self.Zres_joint],j)

                generated_batch['train_gtjoints'][i] = out

                if self.show:
                    visualizer.draw3dskeleton(self.joints_num, self.Zres_joint, out)
                    visualizer.draw3dskeleton_joints(16,19,generated_batch['train_3djoints'][i])
            i = i + 1
            self.currentindex = self.currentindex + 1
            if self.currentindex >= self.datanum:
                random.shuffle(self.filelist)
                self.currentindex = 0
            continue

        return generated_batch


if __name__ == '__main__':

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'

    # train_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/train'
    # valid_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/valid'
    # test_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/test'

    train_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/data'
    valid_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/data'
    test_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/data_test'

    # train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/processed'
    # test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/human3.6m/test'
    # valid_dir = '/home/sicong/detail_data/data'
    # test_dir = '/home/sicong/detail_data/data'

    bg_dir = '/local-scratch2/normal_dataset/bg_dataset'

    meanRgb_dir = '/media/vision-gpu/OLD/humen_depth_data/meanRgb.t7'
    #threadsnum = 10

    # generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir,
    #                                         meanRgb_dir, False, False, 1, data_split_type
    #                                         , dataset_type, 256, 64)
    generator = DataGenerator(train_dir, valid_dir, test_dir, bg_dir, meanRgb_dir, True, True, datatype='detail_data')
    generator._reset_filelist()
    print('reset done! ', generator.datanum, ' files in dataset')
    for index in range(10):
        batch = generator._aux_generator(1)
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
