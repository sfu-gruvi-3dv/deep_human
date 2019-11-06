# -*- coding: utf-8 -*-
"""
author: Kel

original author: Walid Benbihi
"""

import tensorflow as tf

class N_Unet():
    
    def __init__(self, training, nFeat = 256, outDim = 3, name = 'normal_Unet'):
        """
            args:
                nStack     : number of stacks of (hourglass block)
                nFeat      : number of features in each block
                outDim     : number of output dimension (16 for body joints, 15 for 2D segmentation)
                nLow       : how many times of downsampling in hourglass block
                train      : for batch normalization
        """
        self.name = name
        self.nFeat = nFeat
        self.outDim = outDim
        self.train = training
        
    def generate(self, inputs):
        with tf.variable_scope(self.name): 
            with tf.variable_scope('preprocessing'):
                cnv1 = self.lin(inputs, 64, 7, 2, 'SAME', name = '3to64')
                r5 = self.residual(cnv1, 64, name='r5')

            tpnet_out = self.tpnet(r5, self.nFeat)
            ll = self.residual(tpnet_out['normal_out'], self.nFeat, name='ll_res_normal')
            tpnet_out['features'] = ll
            normal_out = self.conv2d(ll, self.outDim, 1, 1, 'SAME', 'final_normal_out')
            tpnet_out['normal_out'] = self.normalize(normal_out)

        return tpnet_out

    def normalize(self, tensor):
        norm = tf.norm(tensor, axis=-1, keepdims=True)
        return tensor / norm


    def down_conv2d(self, inputs, filters, kernel_size = [2, 2], strides = [2, 2], pad = 'VALID', name = 'down_conv2d'):
        with tf.variable_scope(name):
            conv_layer = tf.layers.conv2d(inputs, filters, kernel_size, strides, pad, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv_layer = tf.layers.batch_normalization(conv_layer)
            conv_layer = tf.nn.relu(conv_layer)

            return conv_layer

    def up_con2d(self, inputs, filters, kernel_size = [2, 2], strides = [2, 2], pad = 'SAME', name = 'up_conv2d'):
        with tf.variable_scope(name):
            upsample_input = tf.layers.conv2d_transpose(inputs, filters, kernel_size, strides, pad, use_bias=False, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            upsample_input = tf.layers.batch_normalization(upsample_input)
            upsample_input = tf.nn.relu(upsample_input)

            return  upsample_input

    def conv2d(self, inputs, filters, kernel_size = [2, 2], strides = 1, pad = 'VALID', name = 'conv2d'):
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
        norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, is_training=self.train, activation_fn = tf.nn.relu, scale=True, scope=scope)
        return norm
           
    def lin(self, inputs, filters, kernel_size = 1, strides = 1, pad = 'SAME', name = 'lin'):
        """
           conv -> bn -> relu 
        """
        with tf.variable_scope(name):
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
    
    def tpnet(self, inputs, numOut, name = 'tpnet'):
        """
        Hourglass Block      
        """        
        with tf.variable_scope(name):
            ## normal encoder
            # rough_skip_1 = self.residual(inputs, numOut, name='rough_skip_1')
            normal_skip_1 = self.residual(inputs, 64, name='normal_skip_1') #256x256
            normal_down_1 = self.down_conv2d(inputs, 64, kernel_size=2, strides=2, pad='VALID', name='normal_down_1')

            normal_cov_2 = self.residual(normal_down_1, 64 , name = 'normal_cov_2') #128x128
            normal_skip_2 = self.residual(normal_cov_2, 64, name='normal_skip_2')
            normal_down_2 = self.down_conv2d(normal_cov_2, 64, kernel_size=2, strides=2, pad='VALID', name='normal_down_2')

            normal_cov_3 = self.residual(normal_down_2, 128 , name = 'normal_cov_3')#64x64
            normal_skip_3 = self.residual(normal_cov_3, 128, name='normal_skip_3')
            normal_down_3 = self.down_conv2d( normal_cov_3,128, kernel_size=2, strides=2, pad='VALID', name='normal_down_3')

            normal_cov_4 = self.residual(normal_down_3, 128 , name = 'normal_cov_4')#32x32
            normal_skip_4 = self.residual(normal_cov_4, 128, name='normal_skip_4')
            normal_down_4 = self.down_conv2d( normal_cov_4, 128,  kernel_size=2, strides=2, pad='VALID', name='normal_down_4')

            normal_cov_5 = self.residual(normal_down_4, 256 , name = 'normal_cov_5')#16x16
            normal_skip_5 = self.residual(normal_cov_5, 256, name='normal_skip_5')
            normal_down_5 = self.down_conv2d( normal_cov_5,256,  kernel_size=2, strides=2, pad='VALID', name='normal_down_5')

            last_down = self.residual(normal_down_5, 256 , name = 'last_down') #8x8

            ## The decoder of normal net

            normal_decov_1 = self.residual(last_down, 256, name='normal_decov_1') #16x16
            normal_up_1 = self.up_con2d(normal_decov_1, 256,  kernel_size=2, strides=2, pad='SAME', name='normal_up_1')

            normal_decov_2 = self.residual(tf.add(normal_up_1,normal_skip_5), 128, name='normal_decov_2')#32x32
            normal_up_2 = self.up_con2d(normal_decov_2,128,  kernel_size=2, strides=2, pad='SAME', name='normal_up_2')

            normal_decov_3 = self.residual(tf.add(normal_up_2,normal_skip_4), 128, name='normal_decov_3') #64x64
            normal_up_3 = self.up_con2d(normal_decov_3,128,  kernel_size=2, strides=2, pad='SAME', name='normal_up_3')

            normal_decov_4 = self.residual(tf.add(normal_up_3,normal_skip_3), 64, name='normal_decov_4') #128x128
            normal_up_4 = self.up_con2d(normal_decov_4, 64,  kernel_size=2, strides=2, pad='SAME', name='normal_up_4')

            normal_decov_5 = self.residual(tf.add(normal_up_4,normal_skip_2), 64, name='normal_decov_5') #256x256
            normal_up_5 = self.up_con2d(normal_decov_5,64,  kernel_size=2, strides=2, pad='SAME', name='normal_up_5')

            normal_decov_6 = self.residual(tf.add(normal_up_5,normal_skip_1), 64, name='normal_decov_6') #256x256
            normal_out = self.up_con2d(normal_decov_6,64,  kernel_size=2, strides=2, pad='SAME', name='normal_out')

            output = {}
            output['normal_up_4'] = normal_up_4
            output['normal_up_5'] = normal_up_5
            output['normal_out'] = normal_out

            return output
