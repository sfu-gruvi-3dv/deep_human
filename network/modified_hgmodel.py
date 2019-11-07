# -*- coding: utf-8 -*-
"""
author: Kel

original author: Walid Benbihi
"""

import tensorflow as tf


class ModifiedHgModel():
    def __init__(self, training, nFeat=256, outDim=3, name='single_hourglass'):
        """
            args:
                nFeat      : number of features in each block
                outDim     : number of output dimension (16 for body joints, 15 for 2D segmentation)
                train      : for batch normalization
        """
        self.name = name
        self.nFeat = nFeat
        self.outDim = outDim
        self.train = training

    def generate(self, inputs):
        with tf.variable_scope(self.name):
            with tf.variable_scope('preprocessing'):
                cnv1 = self.lin(inputs, 128, 7, 2, 'SAME', name='256to128')
                # pool = tf.contrib.layers.max_pool2d(r1, [2,2], [2,2], padding= 'VALID')
                #  pool = self.down_conv2d(r1, 128, [2,2], [2,2], pad= 'VALID', name = 'pool')
                #  r4 = self.residual(pool, 128, name='r4')
                r5 = self.residual(cnv1, self.nFeat, name='r5')
                output_feature = self.tpnet(r5, self.nFeat)

                output_feature_256 = self.up_con2d(output_feature, self.nFeat, kernel_size=2, strides=2, pad='SAME', name='output_feature_256')
                output_feature_256_1 = self.residual(output_feature_256, self.nFeat, name='output_feature_256_1')
                basic_feature_256 = self.residual(output_feature_256_1, self.nFeat, name='basic_feature_256')
                residual_feature_256 = self.residual(output_feature_256_1, self.nFeat, name='residual_feature_256')

                outdic = {}
                outdic['depth_feature'] = output_feature_256_1

                with tf.variable_scope('basic_depth'):
                    # basic_feature_256 = self.residual(output_feature_256_1, self.nFeat, name='basic_feature_256')
                    outdic['depth_basic'] = self.conv2d(basic_feature_256, self.outDim, 1, 1, 'SAME', 'depth_basic')
                    # outdic['depth_basic'] = self.conv2d(basic_feature_256, 1, 1, 1, 'SAME', 'depth_basic__')

                with tf.variable_scope('depth_offset'):
                    # residual_feature_256 = self.residual(output_feature_256_1, self.nFeat, name='residual_feature_256')
                    outdic['depth_residual'] = self.conv2d(residual_feature_256, 1, 1, 1, 'SAME', 'depth_residual')

        return outdic



    def down_conv2d(self, inputs, filters, kernel_size=[2, 2], strides=[2, 2], pad='VALID', name='down_conv2d'):
        with tf.variable_scope(name):
            conv_layer = tf.layers.conv2d(inputs, filters, kernel_size, strides, pad, use_bias=False, activation=None,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv_layer = tf.layers.batch_normalization(conv_layer)
            conv_layer = tf.nn.relu(conv_layer)

            return conv_layer

    def up_con2d(self, inputs, filters, kernel_size=[2, 2], strides=[2, 2], pad='SAME', name='up_conv2d'):
        with tf.variable_scope(name):
            upsample_input = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, pad, use_bias=False,
                                                        activation=None,
                                                        kernel_initializer=tf.contrib.layers.xavier_initializer(
                                                            uniform=False))
            upsample_input = tf.layers.batch_normalization(upsample_input)
            upsample_input = tf.nn.relu(upsample_input)

            return upsample_input

    def conv2d(self, inputs, filters, kernel_size=[2, 2], strides=1, pad='VALID', name='conv2d'):
        """
        Typical conv2d layer
        Notice that BN has its own bias term and conv layer before bn does not need bias term.
        However, the bias here will not matter in that case
        """
        with tf.variable_scope(name):
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("b", shape=filters, initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return tf.add(conv, b, 'conv2d_out')

    def bn_relu(self, inputs, scope='bn_relu'):
        """
        bn -> relu
        """
        norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, is_training=self.train, activation_fn=tf.nn.relu,
                                            scale=True, scope=scope)
        return norm

    def lin(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='lin'):
        """
           conv -> bn -> relu
        """
        with tf.variable_scope(name):
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return self.bn_relu(conv, scope='bn_relu')

    def convBlock(self, inputs, numOut, name='convBlock'):
        """
        Convolutional Block
        bn -> relu -> conv(1, 1, numIn, numOut/2)->
        bn -> relu -> conv(3, 3, numOut/2, numOut/2)->
        bn -> relu -> conv(1, 1, numOut/2, numOut)
        """
        with tf.variable_scope(name):
            norm_1 = self.bn_relu(inputs, 'bn_relu_1')
            conv_1 = self.conv2d(norm_1, int(numOut / 2), 1, 1, name='conv_1')

            norm_2 = self.bn_relu(conv_1, 'bn_relu_2')
            conv_2 = self.conv2d(norm_2, int(numOut / 2), 3, 1, 'SAME', name='conv_2')

            norm_3 = self.bn_relu(conv_2, 'bn_relu_3')
            conv_3 = self.conv2d(norm_3, int(numOut), 1, 1, name='conv_3')
            return conv_3

    def skipLayer(self, inputs, numOut, name='skipLayer'):
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

    def residual(self, inputs, numOut, name='residual'):
        """
        Residual Block
        One path to convBlock, the other to skip layer, then sum
        """
        with tf.variable_scope(name):
            convb = self.convBlock(inputs, numOut)
            skip = self.skipLayer(inputs, numOut)
            return tf.add(convb, skip, 'residual_out')

    def tpnet(self, inputs, numOut, name='tpnet'):
        """
        Hourglass Block
        """
        with tf.variable_scope(name):
            with tf.variable_scope('encoder'):
                ## The encoder of detail depth net
                input_skip = self.residual(inputs, numOut, name='input_skip')
                down_1 = self.down_conv2d(inputs, numOut, kernel_size=2, strides=2, pad='VALID', name='down_1') # 64

                cov_1 = self.residual(down_1, numOut, name='cov_1')
                detail_skip_1 = self.residual(cov_1, numOut, name='detail_skip_1')
                down_2 = self.down_conv2d(cov_1, numOut, kernel_size=2, strides=2, pad='VALID', name='down_2') # 32

                cov_2 = self.residual(down_2, numOut, name='cov_2')
                detail_skip_2 = self.residual(cov_2, numOut, name='detail_skip_2')
                down_3 = self.down_conv2d(cov_2, numOut, kernel_size=2, strides=2, pad='VALID', name='down_3') # 16

                cov_3 = self.residual(down_3, numOut, name='cov_3')
                detail_skip_3 = self.residual(cov_3, numOut, name='detail_skip_3')
                down_4 = self.down_conv2d(cov_3, numOut, kernel_size=2, strides=2, pad='VALID', name='down_4') # 8

                cov_4 = self.residual(down_4, numOut, name='cov_4')
                detail_skip_4 = self.residual(cov_4, numOut, name='detail_skip_4')
                down_5_1 = self.down_conv2d(cov_4, numOut, kernel_size=2, strides=2, pad='VALID', name='down_5_1') # 4

                down_5_2 = self.residual(down_5_1, numOut, name='down_5_2') # 4

                down_5_3 = self.residual(down_5_2, numOut, name='down_5_3') # 4


            with tf.variable_scope('decoder'):
                ## The decoder of detail depth net
                detail_deconv_4 = self.up_con2d(down_5_3, numOut, kernel_size=2, strides=2, pad='SAME', name='detail_deconv_4') # 8

                detail_up_4 = self.residual(tf.add(detail_deconv_4, detail_skip_4), numOut, name='detail_up_4')
                detail_deconv_3 = self.up_con2d(detail_up_4, numOut, kernel_size=2, strides=2, pad='SAME', name='detail_deconv_3') # 16

                detail_up_3 = self.residual(tf.add(detail_deconv_3, detail_skip_3), numOut, name='detail_up_3')
                detail_deconv_2 = self.up_con2d(detail_up_3, numOut, kernel_size=2, strides=2, pad='SAME', name='detail_deconv_2') # 32

                detail_up_2 = self.residual(tf.add(detail_deconv_2, detail_skip_2), numOut, name='detail_up_2')
                detail_deconv_1 = self.up_con2d(detail_up_2, numOut, kernel_size=2, strides=2, pad='SAME', name='detail_deconv_1') # 64

                detail_up_1 = self.residual(tf.add(detail_deconv_1, detail_skip_1), numOut, name='detail_up_1')
                detail_deconv_input = self.up_con2d(detail_up_1, numOut, kernel_size=2, strides=2,pad='SAME', name='detail_deconv_input') # 128

                detail_ouput = self.residual(tf.add(detail_deconv_input, input_skip), numOut, name='detail_ouput') # 128



        return detail_ouput