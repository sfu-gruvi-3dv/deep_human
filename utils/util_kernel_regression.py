#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  18/10/18 3:10 PM
#  feitongt
#  util_kernel_regression.py


import numpy as np
##import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def neighbor_index_generator(im_h, im_w, beta, batch_size):
    window_size = beta * 2 + 1
    v_index_np, u_index_np = np.meshgrid(range(im_w), range(im_h), indexing='xy')
    u_index_bucket = np.zeros((1, im_h, im_w, window_size * window_size))
    v_index_bucket = np.zeros((1, im_h, im_w, window_size * window_size))

    bucket_index = 0
    for u_offset in range(-beta, beta + 1):
        for v_offset in range(-beta, beta + 1):
            u_index_shifted_np = u_index_np + u_offset
            v_index_shifted_np = v_index_np + v_offset
            u_index_bucket[0, :, :, bucket_index] = u_index_shifted_np
            v_index_bucket[0, :, :, bucket_index] = v_index_shifted_np
            bucket_index += 1

    u_index_bucket = np.expand_dims(u_index_bucket, axis=-1)
    v_index_bucket = np.expand_dims(v_index_bucket, axis=-1)

    u_index_bucket[u_index_bucket < 0] = 0
    v_index_bucket[v_index_bucket < 0] = 0
    u_index_bucket[u_index_bucket >= im_h] = im_h - 1
    v_index_bucket[v_index_bucket >= im_w] = im_w - 1
    uv_index_bucket = np.concatenate([u_index_bucket, v_index_bucket], axis=-1)
    uv_index_bucket = tf.convert_to_tensor(uv_index_bucket, tf.int32)
    return tf.tile(uv_index_bucket, tf.constant([batch_size, 1, 1, 1, 1]))



def depth_to_pointcloud(depth_map, im_h, im_w, batch_size):
    xv_np, yv_np = np.meshgrid(np.linspace(0, 2, im_w + 1)[:-1], np.linspace(0, 2, im_h + 1)[:-1], indexing='xy')
    xv_np = np.expand_dims(xv_np, axis=-1)
    yv_np = np.expand_dims(yv_np, axis=-1)
    xv_np = np.expand_dims(xv_np, axis=0)
    yv_np = np.expand_dims(yv_np, axis=0)
    xv = tf.convert_to_tensor(xv_np, tf.float32)
    yv = tf.convert_to_tensor(yv_np, tf.float32)
    multiply = tf.constant([batch_size, 1, 1, 1])
    xv = tf.tile(xv, multiply)
    yv = tf.tile(yv, multiply)
    # depth_map = tf.expand_dims(depth_map, axis=-1)
    pointcloud = tf.concat([xv, yv, depth_map], axis = 3)
    return pointcloud


def kernel_regression(batch_size, depth_map, normal_map, beta, delta):
    _, im_h, im_w, _ = depth_map.get_shape().as_list()
    window_size = beta*2 + 1

    pointcloud_map = depth_to_pointcloud(depth_map, im_h, im_w, batch_size)
    uv_indices_bucket = neighbor_index_generator(im_h, im_w, beta, batch_size)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1, 1)), (1, im_h, im_h, window_size*window_size, 1))
    indices = tf.concat([batch_indices, uv_indices_bucket], axis=4)

    neighbor_points = tf.gather_nd(pointcloud_map, indices)
    neighbor_normal = tf.gather_nd(normal_map, indices)
    normal_map_temp = tf.expand_dims(normal_map, axis=3)
    normal_map_temp = tf.tile(normal_map_temp, (1, 1, 1, window_size*window_size, 1))
    normal_similarity = normal_map_temp * neighbor_normal
    normal_similarity = tf.reduce_sum(normal_similarity, axis=4, keep_dims=True)

    delta = tf.constant(delta, name='max_grad')
    zero_tensor = tf.zeros_like(normal_similarity)
    normal_similarity = tf.where(normal_similarity > delta, normal_similarity, zero_tensor)

    pointcloud_map_temp = tf.expand_dims(pointcloud_map, axis=3)
    pointcloud_map_temp = tf.tile(pointcloud_map_temp, (1, 1, 1, window_size*window_size, 1))

    z_ji = tf.reduce_sum(neighbor_points * normal_map_temp, axis=4, keep_dims=True) \
    - tf.reduce_sum(normal_map_temp[:,:,:,:,0:2] * pointcloud_map_temp[:,:,:,:,0:2], axis=4, keep_dims=True)
    z_ji = z_ji / ((tf.expand_dims(normal_map_temp[:,:,:,:,2], axis=-1)))
    z_ji = z_ji * normal_similarity
    z_i = tf.reduce_sum(z_ji, axis=-2)/((tf.reduce_sum(normal_similarity, axis=-2)))
    # z_i = tf.squeeze(z_i, axis=-1)
    return z_i


##############################
###test###
##############################
# im_h = 256
# im_w = 256
# beta = 3
# delta = 0.95
#
# depth_map = np.zeros((5, im_h, im_w))
# normal_map = np.zeros((5, im_h, im_w, 3))
# mask_map = np.zeros((5, im_h, im_w, 1), dtype=bool)
#
# depth_map = tf.convert_to_tensor(depth_map, tf.float32)
# normal_map = tf.convert_to_tensor(normal_map, tf.float32)
# mask_map = tf.convert_to_tensor(mask_map, tf.bool)
#
# kernel_regression(depth_map, normal_map, mask_map, beta, delta)
