# -*- coding: utf-8 -*-
"""
author: Kel
original author: Walid Benbihi
"""

import sys
sys.path.append('../')

import tensorflow as tf

from model import HourglassModel
from normal_Unet import N_Unet
from time import time
import params
import datagen

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.serialization import load_lua
import scipy.io as sio

#from visualizer_2djoint import draw2dskeleton_2
from PIL import Image


generator = datagen.DataGenerator(params.train_dir, params.valid_dir, params.test_dir, params.bg_dir, params.meanRgb_dir, False, True)
meanstd = load_lua(generator.meanRgb_dir)
saved_visual = False
visual = True

if __name__ == '__main__':
    print('Rebuilding Model')
    t_start = time()

#    with tf.name_scope('inputs'):
#        x = tf.placeholder(tf.float32, [None, 256,256,3], name = 'x_train')
#        y = tf.placeholder(tf.float32, [None, 64,64,16], name= 'y_train')
#        phase = tf.placeholder(tf.bool, name="phase")
#
#    with tf.name_scope('model'):
#        hgModel = HourglassModel(phase, params.nStacks, params.nFeat, params.outDim, params.nLow, name = 'hg_pose_2d')
#        output = hgModel.generate(x)

    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x_train')
        y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='y_train')
        mask = tf.placeholder(tf.bool, [None, 256, 256], name='mask')
        phase = tf.placeholder(tf.bool, name="phase")

        w_cos_sim = tf.placeholder(tf.float32, name='w_depth')
        w_norm = tf.placeholder(tf.float32, name='w_grad')
        mask_ext = tf.expand_dims(mask, -1)
        input = tf.concat([x, tf.cast(mask_ext, tf.float32)], axis=-1)

    with tf.variable_scope('normal'):
        # x = tf.placeholder(tf.float32, [None, 256,256,3], name = 'x_train')
        # y = tf.placeholder(tf.float32, [None, 64,64,16], name= 'y_train')
        # phase = tf.placeholder(tf.bool, name="phase")

        #hgModel = HourglassModel(phase, params.nStacks, params.nFeat, params.outDim, params.nLow,
        #                         name='stacked_hourglass')
        #output = hgModel.generate(x)

        normal_model = N_Unet(phase,params.nFeat, params.outDim, name = 'normal_Unet')
        out_dic = normal_model.generate(input)

        output = out_dic['normal_out']

        
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    saver = tf.train.Saver()

    print('Rebuild succeed. Used', str(time()- t_start), 'seconds')


    sess = tf.Session()
    restore_dir = tf.train.latest_checkpoint(params.model_dir)
    if restore_dir:
        saver.restore(sess, restore_dir)
        print('restore succeed')
    else:
        print('Restore failed')
        raise SystemExit

    if visual:
        num_imgs = params.num_imgs
        for i in range(num_imgs):
            train_batch = generator._aux_generator(params.batch_size, 'test', 'normal_dataset', 256, 64)
            img_batch = train_batch['normal_train_img']
            mask_batch = train_batch['normal_train_mask']

            normal_batch = train_batch['normal_train_gtnormal']
            output_ = sess.run(output,feed_dict={x: img_batch, y: normal_batch, mask: mask_batch,w_cos_sim: 10, w_norm: 1, phase: True})
            print('output_.shape: ',output_.shape)
            output_1 = output_
            print('output_1.shape: ',output_1.shape, output_1.dtype)
            print('normal_batch.shape: ', normal_batch.shape, normal_batch.dtype)
            img = img_batch[0]    
            for j in range(3):
                img[:, :, j] = img[:, : ,j] * meanstd['std'][j]
                img[:, :, j] = img[:, : ,j] + meanstd['mean'][j]    
            img = img * 255.0
            img = img.astype(np.uint8)    
            img_out = Image.fromarray(img)
            img_out.show()
            # print(output_1[0, :, :, 0])
            plt.figure()
            plt.imshow(output_1[0,:,:,0], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            plt.figure()
            plt.imshow(output_1[0,:,:,1], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            plt.figure()
            plt.imshow(output_1[0,:,:,2], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            plt.figure()
            plt.imshow(normal_batch[0,:,:,0], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            plt.figure()
            plt.imshow(normal_batch[0,:,:,1], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            plt.figure()
            plt.imshow(normal_batch[0,:,:,2], aspect='auto', cmap=plt.get_cmap('jet'))
            plt.show()
            sio.savemat('np_vector2.mat', {'nx': output_1[0,:,:,0],'ny': output_1[0,:,:,1],'nz': output_1[0,:,:,2],'gtnx':normal_batch[0,:,:,0],'gtny':normal_batch[0,:,:,1],'gtnz':normal_batch[0,:,:,2],'mask':mask_batch[0]})
            #draw2dskeleton(16,19,output_1[0])
            #draw2dskeleton(16,19,heat2d_batch[0])    
            # draw2dskeleton_2(img, 16,19,output_1[0])
            # draw2dskeleton_2(img, 16,19,heat2d_batch[0])
        
        
        
