# -*- coding: utf-8 -*-
"""
author: Kel

original author: Walid Benbihi1111111
"""

import tensorflow as tf


def kaiming(shape, dtype, partition_info=None):
    """Kaiming initialization as described in https://arxiv.org/pdf/1502.01852.pdf

    Args
      shape: dimensions of the tf array to initialize
      dtype: data type of the array
      partition_info: (Optional) info about how the variable is partitioned.
        See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/init_ops.py#L26
        Needed to be used as an initializer.
    Returns
      Tensorflow array with initial weights
    """
    return (tf.truncated_normal(shape, dtype=dtype) * tf.sqrt(2 / float(shape[0])))


class Cascade_Model():

    def __init__(self, training, nFeat = 512, name = 'cross_cascade', reuse = False):
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
        self.train = training

    def generate(self, seg_2d_input, pose_3d_input):
        refined_out = {}
        with tf.variable_scope(self.name):
            #print('feature.shape',seg_2d_input['feature'].shape)
            seg_feature_conv = self.conv2d(seg_2d_input['feature'][1], self.nFeat, 1, 1, name='seg_feature_conv')
            seg_output_conv = self.conv2d(seg_2d_input['out'][1], self.nFeat, 1, 1, name='seg_output_conv')

            joints_feature_conv = self.conv2d(pose_3d_input['feature'][1], self.nFeat, 1, 1, name='joints_feature_conv')
            joints_output_conv = self.conv2d(pose_3d_input['out'][1], self.nFeat, 1, 1, name='joints_ouput_conv')


            refined_seg_input = tf.add_n([seg_feature_conv,seg_output_conv,joints_output_conv,joints_feature_conv])

            # refined_joints_input = tf.add_n([seg_feature_conv,seg_output_conv,joints_output_conv,joints_feature_conv],
            #                              nane='refined_joints_input')
            refined_seg_conv = self.conv2d(refined_seg_input, self.nFeat, 1, 1, name='refined_seg_conv')
            refined_joints_conv = self.conv2d(refined_seg_input, self.nFeat, 1, 1, name='refined_joints_conv')

            seg_refined = self.conv2d(refined_seg_conv, 15, 1, 1, name='seg_refined')
            joints_refined = self.conv2d(refined_joints_conv, 304, 1, 1, name='joints_refined')
            refined_out['seg_refined'] = seg_refined
            refined_out['joints_refined'] = joints_refined
        return refined_out

    def conv2d(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='conv2d'):
        """
        Typical conv2d layer
        Notice that BN has its own bias term and conv layer before bn does not need bias term.
        However, the bias here will not matter in that case
        """
        with tf.variable_scope(name):
            # W = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            b = tf.get_variable("b", shape=filters, initializer=tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return tf.add(conv, b, 'conv2d_out')

    def bn_relu(self, inputs, scope='bn_relu'):
        """
        bn -> relu
        """
        # notice during testing, we need to also set is_training=True because of the huge variance among
        #        bn = tf.layers.batch_normalization(inputs, momentum=0.9, epsilon=1e-5, training=self.train, name = scope)
        #        norm = tf.nn.relu(bn)
        norm = tf.contrib.layers.batch_norm(inputs, 0.9, epsilon=1e-5, is_training=self.train, activation_fn=tf.nn.relu,
                                            scale=True, scope=scope)
        return norm

    def lin(self, inputs, filters, kernel_size=1, strides=1, pad='SAME', name='lin'):
        """
           conv -> bn -> relu
        """
        with tf.variable_scope(name):
            # W = tf.Variable(tf.contrib.layers.xavier_initializer(uniform=False)([kernel_size,kernel_size, inputs.get_shape().as_list()[3], filters]), name= 'weights')
            W = tf.get_variable("W", shape=[kernel_size, kernel_size, inputs.get_shape().as_list()[3], filters],
                                initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            conv = tf.nn.conv2d(inputs, W, [1, strides, strides, 1], padding=pad, data_format='NHWC')
            return self.bn_relu(conv, scope='bn_relu')



class Cam_Model():
    """ A simple Linear+RELU model """
    def __init__(self,
               num_layers,
               linear_size,
               residual,
               batch_norm,
               max_norm,
               batch_size,
               learning_rate,
               dtype=tf.float32):
        """Creates the linear + relu model
        Args
          linear_size: integer. number of units in each layer of the model
          num_layers: integer. number of bilinear blocks in the model
          residual: boolean. Whether to add residual connections
          batch_norm: boolean. Whether to use batch normalization
          max_norm: boolean. Whether to clip weights to a norm of 1
          batch_size: integer. The size of the batches used during training
          learning_rate: float. Learning rate to start with
          summaries_dir: String. Directory where to log progress
          dtype: the data type to use to store internal variables
        """

        self.HUMAN_3D_SIZE = 16 * 3
        self.input_size = self.HUMAN_3D_SIZE
        self.num_layers = 3
        self.isTraining = tf.placeholder(tf.bool, name="isTrainingflag")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.linear_size = linear_size
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        decay_steps = 100000  # empirical
        decay_rate = 0.96  # empirical
        self.learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps, decay_rate)
        self.dtype = dtype

    def generate(self, inputs):
    # === Create the linear + relu combos ===
        with tf.variable_scope( "linear_model" ):

          # === First layer, brings dimensionality up to linear_size ===
          w1 = tf.get_variable( name="w1", initializer=kaiming, shape=[self.input_size, self.linear_size], dtype= self.dtype )
          b1 = tf.get_variable( name="b1", initializer=kaiming, shape=[self.linear_size], dtype= self.dtype )
          w1 = tf.clip_by_norm(w1,1)
          y3 = tf.matmul( inputs, w1 ) + b1
          y3 = tf.layers.batch_normalization(y3,training=self.isTraining, name="batch_normalization")
          y3 = tf.nn.relu( y3 )
          y3 = tf.nn.dropout( y3, self.dropout_keep_prob )

          # === Create multiple bi-linear layers ===
          for idx in range( self.num_layers ):
            y3 = self.two_linear( y3, self.linear_size, True, self.dropout_keep_prob, True, True, self.dtype, idx )

          # === Last linear layer has HUMAN_3D_SIZE in output ===
          w4 = tf.get_variable( name="w4", initializer=kaiming, shape=[self.linear_size, 2], dtype=self.dtype )
          b4 = tf.get_variable( name="b4", initializer=kaiming, shape=[2], dtype=self.dtype )
          w4 = tf.clip_by_norm(w4,1)
          y = tf.matmul(y3, w4) + b4
          return tf.shape(tf.expand_dims(y, -1))


    def two_linear(self, xin, linear_size, residual, dropout_keep_prob, max_norm, batch_norm, dtype, idx):
        """
        Make a bi-linear block with optional residual connection

        Args
          xin: the batch that enters the block
          linear_size: integer. The size of the linear units
          residual: boolean. Whether to add a residual connection
          dropout_keep_prob: float [0,1]. Probability of dropping something out
          max_norm: boolean. Whether to clip weights to 1-norm
          batch_norm: boolean. Whether to do batch normalization
          dtype: type of the weigths. Usually tf.float32
          idx: integer. Number of layer (for naming/scoping)
        Returns
          y: the batch after it leaves the block
        """

        with tf.variable_scope("two_linear_" + str(idx)) as scope:

            input_size = int(xin.get_shape()[1])

            # Linear 1
            w2 = tf.get_variable(name="w2_" + str(idx), initializer=kaiming, shape=[input_size, linear_size], dtype=dtype)
            b2 = tf.get_variable(name="b2_" + str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
            w2 = tf.clip_by_norm(w2, 1) if max_norm else w2
            y = tf.matmul(xin, w2) + b2
            if batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining, name="batch_normalization1" + str(idx))

            y = tf.nn.relu(y)
            y = tf.nn.dropout(y, dropout_keep_prob)

            # Linear 2
            w3 = tf.get_variable(name="w3_" + str(idx), initializer=kaiming, shape=[linear_size, linear_size], dtype=dtype)
            b3 = tf.get_variable(name="b3_" + str(idx), initializer=kaiming, shape=[linear_size], dtype=dtype)
            w3 = tf.clip_by_norm(w3, 1) if max_norm else w3
            y = tf.matmul(y, w3) + b3

            if batch_norm:
                y = tf.layers.batch_normalization(y, training=self.isTraining, name="batch_normalization2" + str(idx))

            y = tf.nn.relu(y)
            y = tf.nn.dropout(y, dropout_keep_prob)

            # Residual every 2 blocks
            y = (xin + y) if residual else y

        return y
