# -*- coding: utf-8 -*-
"""
author: Kel

original author: Walid Benbihi1111111
"""

import tensorflow as tf

class HourglassModel():

    def __init__(self, training, nStack = 2, nFeat = 256, outDim = 16, nLow = 4, name = 'stacked_hourglass', reuse = False):
        """
            args:
                nStack     : number of stacks of (hourglass block)
                nFeat      : number of features in each block
                outDim     : number of output dimension (16 for body joints, 15 for 2D segmentation)
                nLow       : how many times of downsampling in hourglass block
                train      : for batch normalization
        """
        self.nStack = nStack
        self.name = name
        self.nFeat = nFeat
        self.outDim = outDim
        self.train = training
        self.nLow = nLow

    def generate(self, inputs):
        with tf.variable_scope(self.name):
            with tf.variable_scope('preprocessing'):
                cnv1 = self.lin(inputs, 64, 7, 2, 'SAME', name = '256to128')
                r1 = self.residual(cnv1, 128, name='r1')
                pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding= 'VALID')
                r4 = self.residual(pool, 128, name='r4')
                r5 = self.residual(r4, self.nFeat, name='r5')

            out = [None] * self.nStack

            with tf.variable_scope('stacks'):
                inter = r5
                for i in range(self.nStack):
                    with tf.variable_scope('hourglass_' + str(i)):
                        hg = self.hourglass(inter, self.nLow, self.nFeat)
                        ll = self.residual(hg, self.nFeat, name='ll_res')
                        ll = self.lin(ll, self.nFeat, name='ll')
                        tmpOut = self.conv2d(ll, self.outDim, 1, 1, 'SAME','tmpOut')
                        out[i] = tmpOut

                        if i < self.nStack - 1:
                            ll_ = self.conv2d(ll, self.nFeat, name='ll_')
                            tmpOut_ = self.conv2d(tmpOut, self.nFeat, name='tmpOut_')
                            inter = inter + ll_ + tmpOut_

                return tf.stack(out, name = 'output')

    def conv2d(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'SAME', name = 'conv2d'):
        """
        Typical conv2d layer
        Notice that BN has its own bias term and conv layer before bn does not need bias term.
        However, the bias here will not matter in that case
        """
        with tf.variable_scope(name):
            #W = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            W = tf.get_variable("W", shape=[kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("b", shape=filters, initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(inputs, W, [1,strides,strides,1], padding=pad, data_format='NHWC')
            return tf.add(conv, b, 'conv2d_out')

    def bn_relu(self, inputs, scope='bn_relu'):
        """
        bn -> relu
        """
        #notice during testing, we need to also set is_training=True because of the huge variance among
#        bn = tf.layers.batch_normalization(inputs, momentum=0.9, epsilon=1e-5, training=self.train, name = scope)
#        norm = tf.nn.relu(bn)
        norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, is_training=self.train, activation_fn = tf.nn.relu, scale=True, scope=scope)
        return norm

    def lin(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'SAME', name = 'lin'):
        """
           conv -> bn -> relu
        """
        with tf.variable_scope(name):
            #W = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            W = tf.get_variable("W", shape=[kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, W, [1,strides,strides,1], padding=pad, data_format='NHWC')
            return self.bn_relu(conv, scope='bn_relu')

    def convBlock(self, inputs, numOut, name = 'convBlock'):
        """
        Convolutional Block
        bn -> relu -> conv(1, 1, numIn, numOut/2)->
        bn -> relu -> conv(3, 3, numOut/2, numOut/2)->
        bn -> relu -> conv(1, 1, numOut/2, numOut)
        """
        with tf.variable_scope(name):
            norm_1 = self.bn_relu(inputs, 'bn_relu_1')
            conv_1 = self.conv2d(norm_1, int(numOut/2), 1, 1, name='conv_1')

            norm_2 = self.bn_relu(conv_1, 'bn_relu_2')
            conv_2 = self.conv2d(norm_2, int(numOut/2), 3, 1, 'SAME', name='conv_2')

            norm_3 = self.bn_relu(conv_2, 'bn_relu_3')
            conv_3 = self.conv2d(norm_3, int(numOut), 1, 1, name='conv_3')
            return conv_3

    def skipLayer(self, inputs, numOut, name = 'skipLayer'):
        """
        Skip if number of input channel == numOut,
        otherwise use 1x1 conv to remap the channels to a desired number
        """
        with tf.variable_scope(name):
            if inputs.get_shape().as_list()[3] == numOut:
                return inputs
            else:
                conv = self.conv2d(inputs, numOut, 1, 1, 'SAME', name='skipLayer_conv')
                return conv

    def residual(self, inputs, numOut, name = 'residual'):
        """
        Residual Block
        One path to convBlock, the other to skip layer, then sum
        """
        with tf.variable_scope(name):
            convb = self.convBlock(inputs, numOut)
            skip = self.skipLayer(inputs,numOut)
            return tf.add(convb, skip, 'residual_out')

    def hourglass(self, inputs, n, numOut, name = 'hourglass'):
        """
        Hourglass Block
        """
        with tf.variable_scope(name):
            up_1 = self.residual(inputs, numOut, name = 'up1')
            low_ = tf.contrib.layers.max_pool2d(inputs, [2,2],[2,2], 'VALID')
            low_1 = self.residual(low_, numOut, name = 'low1')
            if n > 1:
                low_2 = self.hourglass(low_1, n-1, numOut, name='low2')
            else:
                low_2 = self.residual(low_1, numOut, name='low2')
            low_3 = self.residual(low_2, numOut, name = 'low3')
            up_2 = tf.image.resize_nearest_neighbor(low_3, tf.shape(low_3)[1:3]*2, name= 'upsampling')
            return tf.add(up_1, up_2, 'hourglass_out')
