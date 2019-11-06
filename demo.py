# -*- coding: utf-8 -*-

import argparse
from time import time
from datetime import datetime
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
from tensorflow.contrib.framework.python.framework import checkpoint_utils

import skimage.io as io
from params import params_iccv as params
from network.hourglass_multioutput import HourglassModel
from function import integral_loss
from network.modified_hgmodel import ModifiedHgModel
from utils import util
import scipy.io as sio
import glob

from network.depth_optimization import optimize_depth
from normal.normal_Unet import N_Unet

def masked_huberloss(pred_value, gt_value, mask, delta = 0.3):
    pred_value = tf.boolean_mask(pred_value, mask)
    gt_value = tf.boolean_mask(gt_value, mask)
    loss = tf.losses.huber_loss(gt_value, pred_value, delta=delta)
    return loss

def masked_threholdloss(pred_value, gt_value, mask, delta = 0.3):
    lin = tf.abs(pred_value - gt_value)
    lin = tf.boolean_mask(lin, mask)
    delta = tf.constant(delta, name='max_grad')
    delta_2 = tf.constant(0.005, name='max_grad')

    square = lin
    constant = tf.zeros_like(square)
    error_1 = tf.where(lin < delta, square, constant)
    error_1 = tf.where(lin > delta_2, error_1, constant)

    return tf.reduce_mean(error_1)

def masked_threholdloss_clip(pred_value, gt_value, mask, delta = 0.3):
    lin = tf.abs(pred_value - gt_value)
    lin = tf.boolean_mask(lin, mask)

    square = lin
    error_1 = tf.clip_by_value(square, clip_value_min=0, clip_value_max = delta)

    return tf.reduce_mean(error_1)

def scan_checkpoint_for_vars(checkpoint_path, vars_to_check):
    check_var_list = checkpoint_utils.list_variables(checkpoint_path)
    check_var_list = [x[0] for x in check_var_list]
    check_var_set = set(check_var_list)
    vars_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] in check_var_set]
    vars_not_in_checkpoint = [x for x in vars_to_check if x.name[:x.name.index(":")] not in check_var_set]
    return vars_in_checkpoint, vars_not_in_checkpoint

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('-e', '--eval', action='store_true', help='evaluation')
    argparser.add_argument('-nb', '--num-batches', type=int, default=1, help='number of batches')
    args = argparser.parse_args()
    eval_flag = True
    batch_num = 1

    print('Creating Model')
    t_start = time()

    with tf.variable_scope('joint_train_inputs'):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x_train')
        phase = tf.placeholder(tf.bool, name="phase")
        mask = tf.placeholder(tf.bool, [None, 256, 256], name='y_mask')
        mask_ext = tf.reshape(mask, [-1, 256, 256, 1])
        depth_bilateral_num = tf.placeholder(tf.float32, [None, 256, 256], name='y_depth_bilateral_num')
        depth_bilateral_ext = tf.reshape(depth_bilateral_num, [-1, 256, 256, 1])
        depth_gt_num = tf.placeholder(tf.float32, [None, 256, 256], name='y_depth_raw_num')
        depth_gt_ext = tf.reshape(depth_gt_num, [-1, 256, 256, 1])

    with tf.variable_scope('seg_2d'):
        seg_2d_model = HourglassModel(phase, params.nStacks, params.nFeat, 15, params.nLow,
                                      name='stacked_hourglass')
        seg_2d_output = seg_2d_model.generate(x)


    with tf.variable_scope('pose_3d'):
        pose_3d_model = HourglassModel(phase, params.nStacks, params.nFeat, 304, params.nLow,
                                       name='stacked_hourglass')
        pose_3d_output = pose_3d_model.generate(x)
        w0, h0, d0, _, _, _ = integral_loss.softmax_integral_tensor(pose_3d_output['out'][0], 16, 64, 64, 19)
        w1, h1, d1, _, _, _ = integral_loss.softmax_integral_tensor(pose_3d_output['out'][1], 16, 64, 64, 19)

        pred_joints0 = tf.concat([w0, h0, d0], -1)
        pred_joints1 = tf.concat([w1, h1, d1], -1)

        seg_2d_lowres = seg_2d_output['out'][1]


    seg_2d_rescale = tf.image.resize_images(seg_2d_output['out'][1], [256, 256])
    pose_3d_rescale = tf.image.resize_images(pose_3d_output['out'][1], [256, 256])
    seg_argmax = tf.argmax(seg_2d_rescale, 3)

    zero_tensor = tf.zeros_like(seg_argmax, dtype=tf.int64)

    pref_mask = tf.equal(seg_argmax, zero_tensor, name=None)



    with tf.variable_scope('rough_depth_estimation'):
        rough_depth_model = ModifiedHgModel(phase, params.nFeat, 20, name='single_hourglass')
        depth_dic = rough_depth_model.generate(tf.concat([x, seg_2d_rescale, pose_3d_rescale], axis=-1))

        depth_basic = depth_dic['depth_basic']
        depth_residual = depth_dic['depth_residual']


    with tf.variable_scope('normal_estimation'):
        Normal_model = N_Unet(phase, 256, 3,
                              name='normal_Unet')
        normal_input = tf.concat([x, tf.cast(mask_ext, tf.float32)], axis=-1)
        normal_dic = Normal_model.generate(normal_input)
        rough_normal = normal_dic['normal_out']
        # rough_normal /= (tf.norm(rough_normal, axis=-1, keepdims=True) + 1e-8)
        normal_features = normal_dic['features']

    with tf.variable_scope('output'):
        depth_filter = tf.reshape(tf.range(-9, 11, 1, dtype=tf.float32), [1, 1, 20, 1])
        depth_logits_base = tf.nn.softmax(depth_basic)
        basic_depth = tf.nn.conv2d(depth_logits_base, depth_filter, strides=[1, 1, 1, 1], padding='SAME',
                                   name='depth_basic') * 0.060

        residual_depth = depth_residual/100.0

        refined_depth = basic_depth + residual_depth
        refined_normal = rough_normal

    with tf.variable_scope('kernel_regression'):
        ones_tensor = tf.ones_like(refined_depth)
        zero_tensor = tf.zeros_like(refined_normal)

        beta = 2
        delta = 0.9
        gamma = 0.08
        lamd = 0.2

        regressed_depth = optimize_depth(params.batch_size, refined_depth, refined_depth, refined_normal, beta, delta,
                                         gamma, lamd)
        regressed_depth = optimize_depth(params.batch_size, refined_depth, regressed_depth, refined_normal, beta, delta,
                                         gamma, lamd)
        regressed_depth = optimize_depth(params.batch_size, refined_depth, regressed_depth, refined_normal, beta, delta,
                                         gamma, lamd)
        regressed_depth = optimize_depth(params.batch_size, refined_depth, regressed_depth, refined_normal, beta, delta,
                                         gamma, lamd)
        regressed_depth = optimize_depth(params.batch_size, refined_depth, regressed_depth, refined_normal, beta, delta,
                                         gamma, lamd)

    with tf.variable_scope('loss'):
        loss = masked_huberloss(basic_depth, depth_bilateral_ext, mask_ext, delta=0.20)


    with tf.variable_scope('rmsprop_optimizer'):
        rmsprop = tf.train.RMSPropOptimizer(params.learning_rate)

    with tf.variable_scope('steps'):
        train_steps = tf.Variable(0, trainable=False)

    with tf.variable_scope('minimize'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_rmsprop = rmsprop.minimize(loss, train_steps,
                                             var_list=[tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rough_depth_estimation')])


    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    print('Model generation succeed. Used', str(time() - t_start), 'seconds')

    tf.summary.scalar('loss', loss, collections=['train'])

    merged_summary_op = tf.summary.merge_all('train')

    saver_pose_seg =  tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seg_2d') +
                                     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_3d'))

    saver_depth = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rough_depth_estimation'))

    saver_normal = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='normal_estimation'))


    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        restore_dir = tf.train.latest_checkpoint(params.model_dir)

        if restore_dir:
            global_vars = tf.global_variables()
            vars_in_checkpoint, _ = scan_checkpoint_for_vars(restore_dir, global_vars)
            saver_restore_ckpt = tf.train.Saver(vars_in_checkpoint)
            saver_restore_ckpt.restore(sess, restore_dir)
            normal_dir = tf.train.latest_checkpoint(params.normal_dir)
            saver_normal.restore(sess, normal_dir)
            global_vars = tf.global_variables()
            is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
            not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
            if len(not_initialized_vars):
                sess.run(tf.variables_initializer(not_initialized_vars))
            print('restore succeed')
        else:
            print('restore failed')
            exit(0)

        summary_train = tf.summary.FileWriter(params.log_dir, tf.get_default_graph(),
                                              filename_suffix=params.log_dir)
        t_train_start = datetime.now()


        if eval_flag:

            statics_npy = {}
            basic_depth_err = 0
            detail_depth_err = 0
            regressed_depth_err = 0
            binnum = 40
            hist_basic = np.zeros(binnum)
            hist_detail = np.zeros(binnum)
            hist_regressd = np.zeros(binnum)

            t_train_start = datetime.now()

            # Initialize datalist
            meanstd = util.load_obj(params.meanRgb_dir)
            dataset = glob.glob(params.test_dir+'/*.jpg')
            index = 0
            batch_size = 1
            img_batch = np.zeros((batch_size, 256, 256, 3), dtype=np.float32)
            mask_batch = np.zeros((batch_size,256,256),dtype=np.bool)
            pred_joints_ = np.zeros((batch_size,16,3),dtype=np.int32)
            segmap = np.zeros((batch_size,256,256),dtype=np.int32)
            testimg = np.zeros((256,256,3),dtype = np.float32)
            depth_batch = np.zeros((batch_size,256,256),dtype=np.float32)
            depth_err = 0
            num_data = len(dataset)
            for data in dataset:
                i = 0
                while i < batch_size:
                    img = io.imread(data)
                    for j in range(3):
                        testimg[:, :, j] = np.clip(img[:, :, j].astype(np.float32) / 255.0, 0.0, 1.0)
                        testimg[:, :, j] = testimg[:, :, j] - meanstd['mean'][j]
                        testimg[:, :, j] = testimg[:, :, j] / meanstd['std'][j]
                    img_batch[i] = testimg
                    mask_batch[i] = np.load('/home/sicong/Downloads/de/gtmask_'+segname+'.npy')
                    depth_batch[i] = np.load('/home/sicong/Downloads/de/gtdepth_'+segname+'.npy')

                    plt.figure()
                    plt.imshow(img_batch[i], aspect='auto', cmap=plt.get_cmap('jet'))
                    plt.show()

                    # plt.figure()
                    # plt.imshow(mask_batch[i], aspect='auto', cmap=plt.get_cmap('jet'))
                    # plt.show()
                    #
                    # plt.figure()
                    # plt.imshow(depth_batch[i], aspect='auto', cmap=plt.get_cmap('jet'))
                    # plt.show()
                    i = i+1


                regressed_depth_, refined_depth_, basic_depth_, seg_pred_ ,pred_normal_= sess.run(
                    [regressed_depth, refined_depth, basic_depth, pref_mask,refined_normal],
                    feed_dict={
                        x: img_batch,
                        mask: mask_batch,
                        phase: True})

                refined_depth_ = np.clip(refined_depth_, -0.8, 0.8)
                regressed_depth_ = np.clip(regressed_depth_, -0.8, 0.8)
                refined_depth_ = np.clip(refined_depth_, -0.8, 0.8)

                ori_img = util.restore_rgb(img_batch, params.meanRgb_dir)

                normaldic = {}
                maskdic = {}
                for i in range(params.batch_size):
                    normaldic['Normal_est'] = pred_normal_[i]
                    maskdic['mask'] = mask_batch[i]
                    sio.savemat(os.path.join(params.output_dir, '{}_{}_mask.mat'.format(index, i)),
                                maskdic)
                    sio.savemat(os.path.join(params.output_dir, '{}_{}_normal.mat'.format(index, i)),normaldic)
                    plt.imsave(os.path.join(params.output_dir, '{}_{}_img.png'.format(index, i)), ori_img[i])

                    util.depth2mesh(refined_depth_[i], mask_batch[i],
                                    os.path.join(params.output_dir, '{}_{}_detail_depth'.format(index, i)))
                    util.depth2mesh(basic_depth_[i], mask_batch[i],
                                    os.path.join(params.output_dir, '{}_{}_basic_depth'.format(index, i)))
                    util.depth2mesh(regressed_depth_[i], mask_batch[i],
                                    os.path.join(params.output_dir, '{}_{}_regress_depth'.format(index, i)))
                    util.depth2mesh(depth_batch[i], mask_batch[i],
                                    os.path.join(params.output_dir, '{}_{}_gt_depth'.format(index, i)))

                # static_out_basic = util.new_compute_depth_err_auc(index, basic_depth_, depth_batch, mask_batch,
                #                                                    mask_batch, mask_batch, 'heatmap', 40)
                # static_out_detail = util.new_compute_depth_err_auc(index, refined_depth_, depth_batch, mask_batch,
                #                                                    mask_batch,mask_batch, 'heatmap',40)
                # static_out_regress = util.new_compute_depth_err_auc(index, regressed_depth_, depth_batch, mask_batch,
                #                                                    mask_batch,mask_batch, 'heatmap',40)

                index += 1
                # basic_depth_err += static_out_basic['err']
                # hist_basic += static_out_basic['histmap']
                #
                # detail_depth_err += static_out_detail['err']
                # hist_detail += static_out_detail['histmap']
                #
                # regressed_depth_err += static_out_regress['err']
                # hist_regressd += static_out_regress['histmap']


            # detail_depth_err /= num_data
            # basic_depth_err /= num_data
            # regressed_depth_err /= num_data
            #
            # hist_detail = hist_detail / np.sum(hist_detail)
            #
            # statics_npy['hist_detail'] = hist_detail
            #
            #
            # hist_basic = hist_basic / np.sum(hist_basic)
            #
            # statics_npy['hist_basic'] = hist_basic
            #
            #
            # hist_regressd = hist_regressd / np.sum(hist_regressd)
            #
            # statics_npy['hist_regressed'] = hist_regressd
            # statics_npy['basic_depth_err'] = basic_depth_err
            # statics_npy['detail_depth_err'] = detail_depth_err
            # statics_npy['regressed_depth_err'] = regressed_depth_err
            # sio.savemat('/home/sicong/Downloads/out1/result.mat', statics_npy)
