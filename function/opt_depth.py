#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  23/10/18 7:23 PM
#  feitongt
#  opt_depth.py

import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def persp_depth_opt_stereo(source_img, target_img,
                    source_depth, target_depth,
                    R_mat, T_mat, input_masks, intrinsic_mat,
                    im_w, im_h):
    source_img = source_img
    target_img = target_img
    image_size = tf.shape(source_img)[1]
    source_depth = tf.squeeze(source_depth)

    source_mask = input_masks[0]
    xv_np, yv_np = np.meshgrid(np.linspace(0, im_w, im_w + 1 )[:-1], np.linspace(0, im_h, im_h + 1)[:-1], indexing='xy')
    xv = tf.convert_to_tensor(xv_np, tf.float32)
    yv = tf.convert_to_tensor(yv_np, tf.float32)
    T_mat = tf.expand_dims(T_mat, -1)

    point_x = (xv - intrinsic_mat[0, :, :, 0, 2]) * source_depth / intrinsic_mat[0, :, :, 0, 0]
    point_y = (yv - intrinsic_mat[0, :, :, 1, 2]) * source_depth / intrinsic_mat[0, :, :, 1, 1]

    source_depth = tf.boolean_mask(source_depth, source_mask)
    point_x = tf.boolean_mask(point_x, source_mask)
    point_y = tf.boolean_mask(point_y, source_mask)

    R_mat = tf.boolean_mask(R_mat, source_mask)
    T_mat = tf.boolean_mask(T_mat, source_mask)

    source_points = tf.stack([point_x, point_y, source_depth], -1)
    source_points = tf.expand_dims(source_points, -1)

    deformed_points = tf.matmul(R_mat, source_points) + T_mat

    target_intrinsic = tf.boolean_mask(intrinsic_mat[1], source_mask)


    deformed_points = tf.matmul(target_intrinsic, deformed_points)
    vert_dfm_x = deformed_points[:,0,0]/deformed_points[:,2,0]
    vert_dfm_y = deformed_points[:,1,0]/deformed_points[:,2,0]
    vert_dfm_z = deformed_points[:,2,0]
    dfm_mask_coord = tf.stack([vert_dfm_y, vert_dfm_x, vert_dfm_z], -1)


    #neighbor 1
    vert_dfm_x0 = tf.cast(tf.floor(vert_dfm_x), tf.int32)
    vert_dfm_x1 = tf.cast(tf.ceil(vert_dfm_x), tf.int32)

    vert_dfm_y0 = tf.cast(tf.floor(vert_dfm_y), tf.int32)
    vert_dfm_y1 = tf.cast(tf.ceil(vert_dfm_y), tf.int32)

    idx00 = tf.stack([vert_dfm_y0, vert_dfm_x0], -1)
    idx01 = tf.stack([vert_dfm_y0, vert_dfm_x1], -1)
    idx10 = tf.stack([vert_dfm_y1, vert_dfm_x0], -1)
    idx11 = tf.stack([vert_dfm_y1, vert_dfm_x1], -1)

    I00 = tf.stop_gradient(tf.gather_nd(target_img, idx00))
    I01 = tf.stop_gradient(tf.gather_nd(target_img, idx01))
    I10 = tf.stop_gradient(tf.gather_nd(target_img, idx10))
    I11 = tf.stop_gradient(tf.gather_nd(target_img, idx11))

    vert_target_mask_x = tf.cast(vert_dfm_x, tf.int32)
    vert_target_mask_y = tf.cast(vert_dfm_y, tf.int32)
    dfm_mask_coord = tf.stack([vert_target_mask_y, vert_target_mask_x], -1)
    #print(dfm_mask_coord.get_shape().as_list())
    dfm_and_target_mask = tf.gather_nd(tf.expand_dims(input_masks[1], axis=-1), dfm_mask_coord)
    masked_target_depth = tf.gather_nd(target_depth, dfm_mask_coord)
    masked_target_depth = tf.squeeze(masked_target_depth, axis=-1)

    vert_ori_x = tf.cast(tf.boolean_mask(xv, source_mask), tf.int32)
    vert_ori_y = tf.cast(tf.boolean_mask(yv, source_mask), tf.int32)
    idx_ori = tf.stack([vert_ori_y, vert_ori_x], -1)

    I = tf.gather_nd(source_img, idx_ori)

    tx = tf.expand_dims(vert_dfm_x - tf.floor(vert_dfm_x), -1)
    ty = tf.expand_dims(vert_dfm_y - tf.floor(vert_dfm_y), -1)
    I0x = (1.0 - tx) * I00 + tx * I01
    I1x = (1.0 - tx) * I10 + tx * I11
    Ixx = (1.0 - ty) * I0x + ty * I1x

    indices = tf.cast(idx_ori, tf.int32)
    shape = tf.constant([256,256,3])
    warpimg = tf.scatter_nd(indices,Ixx,shape)

    intensity_diff = tf.abs(I - Ixx)

    depth_consist_mask = tf.less(tf.abs(masked_target_depth-vert_dfm_z), 0.50 * tf.ones_like(masked_target_depth))
    final_mask = tf.logical_and(depth_consist_mask, tf.squeeze(dfm_and_target_mask, axis=-1))
    indices = tf.where(final_mask)

    masked_intensity_diff = tf.gather_nd(intensity_diff, indices)
    masked_intensity_diff = tf.reduce_sum(masked_intensity_diff,axis=-1)

    masked_Ixx = tf.reduce_sum(tf.gather_nd(Ixx, indices),axis=-1)
    zeros_tensor = tf.zeros_like(masked_intensity_diff, dtype=tf.float32)

    threshold_1 = tf.constant(400, name='threshold', dtype=tf.float32)
    threshold_2 = tf.constant(0, name='threshold', dtype=tf.float32)

    # masked_intensity_diff_out = masked_intensity_diff/tf.stop_gradient(masked_Ixx)
    masked_intensity_diff_out = masked_intensity_diff

    masked_intensity_diff_out = tf.where(masked_intensity_diff < threshold_1, masked_intensity_diff_out, zeros_tensor)
    masked_intensity_diff_out = tf.where(masked_intensity_diff > threshold_2, masked_intensity_diff_out, zeros_tensor)

    loss = tf.reduce_mean(masked_intensity_diff_out)
    # loss = tf.reduce_mean(source_depth)

    # variable for debug
    out_debug_dic = {}
    out_debug_dic['source_img'] = source_img
    out_debug_dic['target_img'] = target_img
    out_debug_dic['idx_ori'] = idx_ori
    out_debug_dic['idx00'] = idx00
    out_debug_dic['idx_dst'] = tf.stack([vert_dfm_y, vert_dfm_x], -1)
    out_debug_dic['warpimg'] = warpimg

    out_debug_dic['depth_consist_mask'] = depth_consist_mask
    out_debug_dic['dfm_and_target_mask'] = dfm_and_target_mask
    out_debug_dic['final_mask'] = final_mask
    out_debug_dic['image_size'] = tf.cast(image_size, tf.float32)
    out_debug_dic['deformed_points'] = deformed_points
    out_debug_dic['dfm_mask_coord'] = dfm_mask_coord
    out_debug_dic['masked_intensity_diff'] = masked_intensity_diff
    out_debug_dic['masked_target_depth'] = masked_target_depth
    out_debug_dic['vert_dfm_z'] = vert_dfm_z


    return loss, out_debug_dic

def persp_depth_opt(source_img, target_img,
                    source_depth, target_depth,
                    R_mat, T_mat, input_masks, intrinsic_mat,
                    im_w, im_h):
    source_img = source_img
    target_img = target_img
    image_size = tf.shape(source_img)[1]
    source_depth = tf.squeeze(source_depth)

    source_mask = input_masks[0]
    xv_np, yv_np = np.meshgrid(np.linspace(0, im_w, im_w + 1 )[:-1], np.linspace(0, im_h, im_h + 1)[:-1], indexing='xy')
    xv = tf.convert_to_tensor(xv_np, tf.float32)
    yv = tf.convert_to_tensor(yv_np, tf.float32)
    T_mat = tf.expand_dims(T_mat, -1)

    point_x = (xv - intrinsic_mat[:, :, 0, 2]) * source_depth / intrinsic_mat[:, :, 0, 0]
    point_y = (yv - intrinsic_mat[:, :, 1, 2]) * source_depth / intrinsic_mat[:, :, 1, 1]

    source_depth = tf.boolean_mask(source_depth, source_mask)
    point_x = tf.boolean_mask(point_x, source_mask)
    point_y = tf.boolean_mask(point_y, source_mask)

    R_mat = tf.boolean_mask(R_mat, source_mask)
    T_mat = tf.boolean_mask(T_mat, source_mask)

    source_points = tf.stack([point_x, point_y, source_depth], -1)
    source_points = tf.expand_dims(source_points, -1)

    deformed_points = tf.matmul(R_mat, source_points) + T_mat

    intrinsic_mat = tf.boolean_mask(intrinsic_mat, source_mask)


    deformed_points = tf.matmul(intrinsic_mat, deformed_points)
    vert_dfm_x = deformed_points[:,0,0]/deformed_points[:,2,0]
    vert_dfm_y = deformed_points[:,1,0]/deformed_points[:,2,0]
    vert_dfm_z = deformed_points[:,2,0]
    dfm_mask_coord = tf.stack([vert_dfm_y, vert_dfm_x, vert_dfm_z], -1)


    #neighbor 1
    vert_dfm_x0 = tf.cast(tf.floor(vert_dfm_x), tf.int32)
    vert_dfm_x1 = tf.cast(tf.ceil(vert_dfm_x), tf.int32)

    vert_dfm_y0 = tf.cast(tf.floor(vert_dfm_y), tf.int32)
    vert_dfm_y1 = tf.cast(tf.ceil(vert_dfm_y), tf.int32)

    idx00 = tf.stack([vert_dfm_y0, vert_dfm_x0], -1)
    idx01 = tf.stack([vert_dfm_y0, vert_dfm_x1], -1)
    idx10 = tf.stack([vert_dfm_y1, vert_dfm_x0], -1)
    idx11 = tf.stack([vert_dfm_y1, vert_dfm_x1], -1)

    I00 = tf.stop_gradient(tf.gather_nd(target_img, idx00))
    I01 = tf.stop_gradient(tf.gather_nd(target_img, idx01))
    I10 = tf.stop_gradient(tf.gather_nd(target_img, idx10))
    I11 = tf.stop_gradient(tf.gather_nd(target_img, idx11))

    vert_target_mask_x = tf.cast(vert_dfm_x, tf.int32)
    vert_target_mask_y = tf.cast(vert_dfm_y, tf.int32)
    dfm_mask_coord = tf.stack([vert_target_mask_y, vert_target_mask_x], -1)
    #print(dfm_mask_coord.get_shape().as_list())
    dfm_and_target_mask = tf.gather_nd(tf.expand_dims(input_masks[1], axis=-1), dfm_mask_coord)
    masked_target_depth = tf.gather_nd(target_depth, dfm_mask_coord)
    masked_target_depth = tf.squeeze(masked_target_depth, axis=-1)

    vert_ori_x = tf.cast(tf.boolean_mask(xv, source_mask), tf.int32)
    vert_ori_y = tf.cast(tf.boolean_mask(yv, source_mask), tf.int32)
    idx_ori = tf.stack([vert_ori_y, vert_ori_x], -1)

    I = tf.gather_nd(source_img, idx_ori)

    tx = tf.expand_dims(vert_dfm_x - tf.floor(vert_dfm_x), -1)
    ty = tf.expand_dims(vert_dfm_y - tf.floor(vert_dfm_y), -1)
    I0x = (1.0 - tx) * I00 + tx * I01
    I1x = (1.0 - tx) * I10 + tx * I11
    Ixx = (1.0 - ty) * I0x + ty * I1x

    indices = tf.cast(idx_ori, tf.int32)
    shape = tf.constant([256,256,3])
    warpimg = tf.scatter_nd(indices,Ixx,shape)

    intensity_diff = tf.abs(I - Ixx)

    depth_consist_mask = tf.less(tf.abs(masked_target_depth-vert_dfm_z), 0.50 * tf.ones_like(masked_target_depth))
    final_mask = tf.logical_and(depth_consist_mask, tf.squeeze(dfm_and_target_mask, axis=-1))
    indices = tf.where(final_mask)

    masked_intensity_diff = tf.gather_nd(intensity_diff, indices)
    masked_intensity_diff = tf.reduce_sum(masked_intensity_diff,axis=-1)

    masked_Ixx = tf.reduce_sum(tf.gather_nd(Ixx, indices),axis=-1)
    zeros_tensor = tf.zeros_like(masked_intensity_diff, dtype=tf.float32)

    threshold_1 = tf.constant(400, name='threshold', dtype=tf.float32)
    threshold_2 = tf.constant(0, name='threshold', dtype=tf.float32)

    # masked_intensity_diff_out = masked_intensity_diff/tf.stop_gradient(masked_Ixx)
    masked_intensity_diff_out = masked_intensity_diff

    masked_intensity_diff_out = tf.where(masked_intensity_diff < threshold_1, masked_intensity_diff_out, zeros_tensor)
    masked_intensity_diff_out = tf.where(masked_intensity_diff > threshold_2, masked_intensity_diff_out, zeros_tensor)

    loss = tf.reduce_mean(masked_intensity_diff_out)
    # loss = tf.reduce_mean(source_depth)

    # variable for debug
    out_debug_dic = {}
    out_debug_dic['source_img'] = source_img
    out_debug_dic['target_img'] = target_img
    out_debug_dic['idx_ori'] = idx_ori
    out_debug_dic['idx00'] = idx00
    out_debug_dic['idx_dst'] = tf.stack([vert_dfm_y, vert_dfm_x], -1)
    out_debug_dic['warpimg'] = warpimg

    out_debug_dic['depth_consist_mask'] = depth_consist_mask
    out_debug_dic['dfm_and_target_mask'] = dfm_and_target_mask
    out_debug_dic['final_mask'] = final_mask
    out_debug_dic['image_size'] = tf.cast(image_size, tf.float32)
    out_debug_dic['deformed_points'] = deformed_points
    out_debug_dic['dfm_mask_coord'] = dfm_mask_coord
    out_debug_dic['masked_intensity_diff'] = masked_intensity_diff
    out_debug_dic['masked_target_depth'] = masked_target_depth
    out_debug_dic['vert_dfm_z'] = vert_dfm_z


    return loss, out_debug_dic


def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy


def compute_smooth_loss(disp, img, mask):
    disp_gradients_x = gradient_x(disp)
    disp_gradients_y = gradient_y(disp)

    image_gradients_x = gradient_x(img)
    image_gradients_y = gradient_y(img)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * weights_x * tf.to_float(mask[:,:,:-1,:])
    smoothness_y = disp_gradients_y * weights_y * tf.to_float(mask[:,:-1,:,:])

    return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))
