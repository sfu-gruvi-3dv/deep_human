#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  27/09/18 10:26 PM
#  feitongt
#  integral_loss.py
import tensorflow as tf

def generate_3d_integral_preds_tensor(heatmaps, num_joints, h_dim, w_dim, d_dim):
    heatmaps = tf.reshape(heatmaps,[tf.shape(heatmaps)[0], num_joints, d_dim, h_dim, w_dim])
    heatmaps = tf.transpose(heatmaps, [0, 1, 3, 4, 2])

    accu_h = tf.reduce_sum(heatmaps, axis=4)
    accu_h_p = tf.reduce_sum(accu_h, axis=3)

    accu_w = tf.reduce_sum(heatmaps, axis=2)
    accu_w_p = tf.reduce_sum(accu_w, axis=3)

    accu_d = tf.reduce_sum(heatmaps, axis=2)
    accu_d_p = tf.reduce_sum(accu_d, axis=2)

    accu_h_filter = tf.reshape(tf.range(0, h_dim, dtype=tf.float32), [1, h_dim, 1])
    accu_h = tf.nn.conv1d(accu_h_p, accu_h_filter, stride=1, padding='SAME',name='accu_h')

    accu_w_filter = tf.reshape(tf.range(0, w_dim, dtype=tf.float32), [1, w_dim, 1])
    accu_w = tf.nn.conv1d(accu_w_p, accu_w_filter, stride=1, padding='SAME',name='accu_w')

    accu_d_filter = tf.reshape(tf.range(0, 19, dtype=tf.float32), [1, d_dim,1])
    accu_d = tf.nn.conv1d(accu_d_p, accu_d_filter, stride=1, padding='SAME',name='accu_d')

    return accu_h, accu_w, accu_d, accu_h_p, accu_w_p, accu_d_p

def softmax_integral_tensor(preds, num_joints, height, width, depth):
    preds = tf.transpose(preds, [0, 3, 1, 2])
    preds = tf.reshape(preds, [tf.shape(preds)[0], num_joints, depth*height*width])
    preds = tf.nn.softmax(preds)
    h, w, d, accu_h_p, accu_w_p, accu_d_p = generate_3d_integral_preds_tensor(preds,  num_joints, height, width, depth)

    return w, h, d, accu_h_p, accu_w_p, accu_d_p