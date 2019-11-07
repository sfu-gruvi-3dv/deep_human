# -*- coding: utf-8 -*-
"""
author: Sicong, Feitong, Kel

"""
import tensorflow as tf
import params.params_iccv as params
import utils.util as util
from network.hourglass_multioutput import HourglassModel
from network.Refine_model import Cascade_Model
import function.integral_loss as integral_loss
import numpy as np
from network.modified_hgmodel import ModifiedHgModel
import matplotlib.pyplot as plt
import cv2,glob,os
import skimage.io as io
from normal.normal_Unet import N_Unet
from network.depth_optimization import optimize_depth

if __name__ == '__main__':
    print('Creating Model')
    with tf.variable_scope('joint_train_inputs'):
        x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x_train')
        phase = tf.placeholder(tf.bool, name="phase")

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

        pred_joints1 = tf.concat([w1, h1, d1], -1)
        pred_joints0 = tf.concat([w0, h0, d0], -1)

    with tf.variable_scope('cross_cascade_refinement'):
        cross_cascade_model = Cascade_Model(phase, 512, name='Cascade_Model')
        pose_seg_refined = cross_cascade_model.generate(seg_2d_output, pose_3d_output)
        w2, h2, d2, _, _, _ = integral_loss.softmax_integral_tensor(pose_seg_refined['joints_refined'], 16, 64, 64, 19)
        pred_joints_refined = tf.concat([w2, h2, d2], -1)

    seg_2d_rescale = tf.image.resize_images(pose_seg_refined['seg_refined'], [256, 256])
    pose_3d_rescale = tf.image.resize_images(pose_seg_refined['joints_refined'], [256, 256])


    with tf.variable_scope('rough_depth_estimation'):
        rough_depth_model = ModifiedHgModel(phase, params.nFeat, 20, name='single_hourglass')
        depth_dic = rough_depth_model.generate(tf.concat([x, seg_2d_rescale, pose_3d_rescale], axis=-1))

        depth_basic = depth_dic['depth_basic']
        depth_residual = depth_dic['depth_residual']

    with tf.variable_scope('normal_estimation'):

        seg_pred = tf.argmax(seg_2d_rescale, axis=-1)
        zero_tensor = tf.zeros_like(seg_pred)
        seg_mask = tf.not_equal(seg_pred, zero_tensor)
        mask_ext = tf.reshape(seg_mask, [-1, 256, 256, 1])

        Normal_model = N_Unet(phase, 256, 3,
                              name='normal_Unet')
        normal_input = tf.concat([x, tf.cast(mask_ext, tf.float32)], axis=-1)
        normal_dic = Normal_model.generate(normal_input)
        rough_normal = normal_dic['normal_out']


    with tf.variable_scope('rough_depth'):
        depth_filter = tf.reshape(tf.range(-9, 11, 1, dtype=tf.float32), [1, 1, 20, 1])

        depth_logits_final = tf.nn.softmax(depth_basic)
        final_depth = tf.nn.conv2d(depth_logits_final, depth_filter, strides=[1, 1, 1, 1], padding='SAME',
                                   name='depth_1') * 0.060

        final_depth_offset = depth_residual / 100.0

        pred_basic_depth =  final_depth
        pred_detail_depth = final_depth + final_depth_offset

    with tf.variable_scope('depth_refinement'):
        refined_depth = pred_detail_depth
        refined_normal = rough_normal
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

    init = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())

    saver_pose_seg_depth = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='seg_2d') +
                                    tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose_3d')+
                                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cross_cascade_refinement')+
                                          tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rough_depth_estimation'))

    saver_normal = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='normal_estimation'))
    saver = tf.train.Saver()
    sess = tf.Session()

    restore_dir = tf.train.latest_checkpoint(params.model_dir)
    if restore_dir:
        saver_pose_seg_depth.restore(sess, restore_dir)
        print('restore pose_seg_depth succeed')
    else:
        print('Restore pose_seg_depth failed')
        raise SystemExit

    restore_dir = tf.train.latest_checkpoint(params.normal_dir)
    if restore_dir:
        saver_normal.restore(sess, restore_dir)
        print('restore normal succeed')
    else:
        print('Restore normal failed')
        raise SystemExit

    # Initialize datalist
    meanstd = util.load_obj(params.meanRgb_dir)
    dataset = glob.glob(params.test_dir + '/*.jpg')
    img_batch = np.zeros((params.batch_size, 256, 256, 3), dtype=np.float32)
    mask_batch = np.zeros((params.batch_size, 256, 256), dtype=np.bool)
    segmap = np.zeros((params.batch_size, 256, 256), dtype=np.int32)
    testimg = np.zeros((256, 256, 3), dtype=np.float32)

    for imgname in dataset:
        i = 0
        while i < params.batch_size:
            img = io.imread(imgname)
            index = imgname.split('/')[-1].split('_')[0]

            for j in range(3):
                testimg[:, :, j] = np.clip(img[:, :, j].astype(np.float32) / 255.0, 0.0, 1.0)
                testimg[:, :, j] = testimg[:, :, j] - meanstd['mean'][j]
                testimg[:, :, j] = testimg[:, :, j] / meanstd['std'][j]

            img_batch[i] = testimg
            depth_gt = np.load(params.test_dir + '/' + index + '_depth.npy')
            depth_gt = cv2.resize(depth_gt, (256, 256), interpolation=cv2.INTER_NEAREST)
            mask_batch[i] = depth_gt > 1e-3
            i = i + 1

            pred_basic_depth_, pred_detail_depth_,regressed_depth_ = sess.run(
            [pred_basic_depth, pred_detail_depth,regressed_depth],
            feed_dict={
                x: img_batch,
                phase: True})

        basic_depth_ = np.clip(pred_basic_depth_, -0.8, 0.8)
        regressed_depth_ = np.clip(pred_detail_depth_, -0.8, 0.8)

        ori_img = util.restore_rgb(img_batch, params.meanRgb_dir)

        normaldic = {}
        maskdic = {}
        for i in range(params.batch_size):
            util.depth2mesh(basic_depth_[i], mask_batch[i],
                            os.path.join(params.output_dir, index + '_' + str(i) + '_basic_depth'))
            util.depth2mesh(pred_detail_depth_[i], mask_batch[i],
                            os.path.join(params.output_dir, index + '_' + str(i) + '_detail_depth'))
            util.depth2mesh(regressed_depth_[i], mask_batch[i],
                            os.path.join(params.output_dir, index + '_' + str(i) + '_regress_depth'))


