# -*- coding: utf-8 -*-
"""
Author: Kel

original author: Walid Benbihi
"""

import sys
sys.path.append('../')

import tensorflow as tf
import numpy as np
from hourglass_multioutput import HourglassModel
from normal_Unet import N_Unet
from time import time, strftime
from datetime import datetime
import params
from datagen_single_thread import DataGenerator
import argparse
from skimage import io
import matplotlib.pyplot as plt
import math
from normal_util import apply_mask, transform_normal, draw_normal_sphere, normalize, depth_to_normal
import seaborn as sns
import os.path as osp
import os






if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-e', '--eval', action='store_true', help='evaluation mode')
    arg_parser.add_argument('-m', '--modeldir', help='model dir')
    arg_parser.add_argument('-v', '--visualize', action='store_true', help='visualize results')
    arg_parser.add_argument('-sf', '--savefig', help='save figure dir')
    arg_parser.add_argument('-nb', '--num-batches', type=int, default=1, help='number of batches')
    arg_parser.add_argument('-t', '--test-type', choices=['test', 'real'])
    args = arg_parser.parse_args()


    print('Creating Model')
    t_start = time()

    with tf.name_scope('inputs'):
       x = tf.placeholder(tf.float32, [None, 256,256,3], name = 'x_train')
       y = tf.placeholder(tf.float32, [None, 256,256,3], name= 'y_train')
       mask = tf.placeholder(tf.bool, [None, 256,256], name= 'mask')
       phase = tf.placeholder(tf.bool, name="phase")

       # w_cos_sim = tf.placeholder(tf.float32, name='w_depth')
       # w_norm = tf.placeholder(tf.float32, name='w_grad')
       mask_ext = tf.expand_dims(mask,-1)
       input = tf.concat([x, tf.cast(mask_ext, tf.float32)],axis = -1)
    with tf.variable_scope('normal_estimation'):

        # hgModel = HourglassModel(phase, params.nStacks, params.nFeat, params.outDim, params.nLow, name = 'stacked_hourglass')
        # output = hgModel(x)['out']
        normal_model = N_Unet(phase, params.nFeat, params.outDim, name = 'normal_Unet')
        out_dic = normal_model.generate(input)
        output = out_dic['normal_out']


    with tf.variable_scope('loss'):
        elem_multi = tf.multiply(output, y)
        cross_1 = tf.reduce_sum(elem_multi, axis=3)
        # output_2 = tf.multiply(output[1], y)
        # cross_2 = tf.reduce_sum(output_2, axis=3)
        norm_1 = tf.norm(output, ord=2, axis=3)
        # norm_2 = tf.norm(output_2, ord=2, axis=3)
        cross_1_masked = tf.boolean_mask(cross_1, mask)
        # cross_2_masked = tf.boolean_mask(cross_2, mask)
        norm_1_masked = tf.boolean_mask(norm_1, mask)
        # norm_2_masked = tf.boolean_mask(norm_2, mask)
        loss = tf.reduce_mean(tf.acos(cross_1_masked/(norm_1_masked+0.00001))/np.pi * 180)
        # loss_2_cos_sim = tf.reduce_mean(cross_2_masked/(norm_2_masked+0.00001)/3.1415926)
        # loss_1_reg = tf.reduce_mean(tf.abs(norm_1_masked-1))
        # loss_2_reg = tf.reduce_mean(tf.abs(norm_2_masked-1))

        # loss = w_cos_sim*(loss_1_cos_sim)+w_norm*(loss_1_reg)

    with tf.variable_scope('rmsprop_optimizer'):
        rmsprop = tf.train.RMSPropOptimizer(params.learning_rate)

    with tf.variable_scope('steps'):
        train_steps = tf.Variable(0, trainable=False)

    with tf.variable_scope('minimize'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_rmsprop = rmsprop.minimize(loss, train_steps)
        
    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    print('Model generation succeed. Used', str(time() - t_start), 'seconds')

    # tf.summary.scalar('loss_1', loss_1_cos_sim, collections = ['train'])
    # tf.summary.scalar('loss_2', loss_1_reg, collections = ['train'])
    tf.summary.scalar('loss', loss, collections = ['train'])
        
    merged_summary_op = tf.summary.merge_all('train')
    saver = tf.train.Saver()

    with tf.Session() as sess:
        restore_dir = tf.train.latest_checkpoint(args.modeldir)
        if restore_dir:
            saver.restore(sess, restore_dir)
            print('restore succeed')
        else:
            sess.run(init)
            print('Session initilized')

        if args.eval:
            print('Test started...')
            test_start = datetime.now()

            if args.test_type == 'test':
                test_dir = params.test_dir
            elif args.test_type == 'real':
                test_dir = params.real_data_dir
            generator = DataGenerator(params.train_dir, params.valid_dir, test_dir, params.bg_dir,
                                              params.meanRgb_dir, False, True)
            if args.test_type == 'test':
                generator._reset_filelist('normal_dataset', 'test')
            elif args.test_type == 'real':
                generator._reset_filelist('detail_data', 'test')

            if args.savefig:
                if not osp.exists(args.savefig):
                    os.mkdir(args.savefig)

            acc_cossim = 0
            acc_ang_diff = 0
            for bi in range(args.num_batches):
                if bi % 10 == 0:
                    print('Processing batch {}'.format(bi))
                if args.test_type == 'test':
                    batch = generator._aux_generator(params.batch_size, 'test', 'normal_dataset', 256, 64)
                elif args.test_type == 'real':
                    batch = generator._aux_generator(params.batch_size, 'test', 'detail_data', 256, 64)

                img_batch = batch['train_img']
                mask_batch = batch['train_mask']
                depth_batch = batch['train_gtdepthre']

                masked_img = apply_mask(img_batch, mask_batch)

                mask_ext = np.expand_dims(mask_batch, -1)
                mask_ext = np.repeat(mask_ext, 3, -1)
                # if osp.normpath(args.modeldir) == osp.normpath('rgb_model'):
                #     features = img_batch
                # elif osp.normpath(args.modeldir) == osp.normpath('grey_grad_model'):
                #     grey = 0.2126 * img_batch[:, :, :, 0] + 0.7152 * img_batch[:, :, :, 1] + 0.0722 * img_batch[:, :, :, 2]
                #     dy, dx = np.gradient(grey, axis=(1, 2))
                #     features = np.stack((dy, dx, grey), axis=-1)

                out_normal = sess.run(output, feed_dict={x: img_batch, mask: mask_batch, phase: False})
                out_normal = normalize(out_normal)
                out_normal = apply_mask(out_normal, mask_batch)
                out_rgbnormal = transform_normal(out_normal)
                out_rgbnormal = apply_mask(out_rgbnormal, mask_batch).astype(np.uint8)
                normal_sphere_r = 25
                out_rgbnormal[:, :2*normal_sphere_r, -2*normal_sphere_r:, :] = draw_normal_sphere(out_normal.shape[0], normal_sphere_r)

                gtnormal = batch['gtnormal']
                # gtnormal = depth_to_normal(depth_batch)
                gtnormal = apply_mask(gtnormal, mask_batch)
                gtrgbnormal = transform_normal(gtnormal)
                gtrgbnormal = apply_mask(gtrgbnormal, mask_batch).astype(np.uint8)
                gtrgbnormal[:, :2 * normal_sphere_r, -2 * normal_sphere_r:, :] = draw_normal_sphere(out_normal.shape[0], normal_sphere_r)

                cossim = np.sum(out_normal * gtnormal, axis=-1)
                cossim = apply_mask(cossim, mask_batch)
                cossim = np.clip(cossim, -1, 1)
                if np.any(cossim > 1) or np.any(cossim < -1):
                    print(cossim[cossim > 1], cossim[cossim < -1])
                acc_cossim += np.mean(cossim[mask_batch])

                ang_diff = np.arccos(cossim) / np.pi * 180
                ang_diff = apply_mask(ang_diff, mask_batch)
                acc_ang_diff += np.mean(ang_diff[mask_batch])

                plt.rcParams['figure.figsize'] = (10, 6)
                if args.visualize or args.savefig:
                    plt.figure()
                    plt.axis('off')
                    for i in range(params.batch_size):
                        plt.subplot(params.batch_size, 3, i*3+1)
                        plt.imshow(out_rgbnormal[i])
                        plt.axis('off')
                        if i == 0:
                            plt.title('Output normal')

                        plt.subplot(params.batch_size, 3, i*3+2)
                        plt.imshow(gtrgbnormal[i])
                        plt.axis('off')
                        if i == 0:
                            plt.title('GT normal')

                        plt.subplot(params.batch_size, 3, i*3+3)
                        sns.heatmap(ang_diff[i], mask=~mask_batch[i], cmap='YlOrRd')
                        plt.axis('off')
                        if i == 0:
                            plt.title('Angular difference')

                    plt.suptitle('visualization of predicted normal, GT normal and angular difference using {}'.format(osp.basename(osp.normpath(args.modeldir))))
                    if args.savefig:
                        plt.savefig(osp.join(args.savefig, str(bi)+'.png'))

            test_end = datetime.now()
            print('Test completed after {}'.format(test_end - test_start))
            if args.visualize:
                plt.show()

            result = 'Average angular difference using {} on {} : {}\n'.format(osp.basename(osp.normpath(args.modeldir)),
                                                                              osp.basename(osp.normpath(test_dir)), acc_ang_diff / args.num_batches)
            print(result)
            # with open('result.txt', 'a') as f:
            #     f.write(result)

        else:
            generator = DataGenerator(params.train_dir, params.valid_dir, params.test_dir, params.bg_dir,
                                              params.meanRgb_dir, False, True)
            generator._reset_filelist('detail_data', 'train')

            summary_train = tf.summary.FileWriter(params.log_dir , tf.get_default_graph())
            t_train_start = time()

            if not osp.exists(args.modeldir):
                os.mkdir(args.modeldir)

            print('Start training')
            for epoch in range(params.nEpochs):
                print('========Training Epoch: ', (epoch + 1))
                for i in range(params.iter_by_epoch):
                    #img_batch, heat2d_batch = generator._aux_generator(params.batch_size, 'train')
                    train_batch = generator._aux_generator(params.batch_size, 'train', 'detail_data', 256, 64)
                    img_batch = train_batch['train_img']
                    mask_batch = train_batch['train_mask']
                    normal_batch = train_batch['gtnormal']
                    # grey =  0.2126*img_batch[:, :, :, 0] + 0.7152*img_batch[:, :, :, 1] + 0.0722*img_batch[:, :, :, 2]
                    # dy, dx = np.gradient(grey, axis=(1, 2))
                    # features = np.stack((dy, dx, grey), axis=-1)
                    _, summary, glb_step, ls = sess.run([train_rmsprop, merged_summary_op, train_steps, loss],
                                                    feed_dict={x:img_batch, y: normal_batch, mask: mask_batch, phase: True})

                    if glb_step % 100 == 0:
                        print("step:",glb_step, " loss: ", ls)

                    if glb_step % 1000 == 0:
                        summary_train.add_summary(summary, glb_step)

                    if (glb_step % params.step_to_save == 0) and (glb_step>0):
                        saver.save(sess, args.modeldir, global_step=train_steps)

        

        
        
