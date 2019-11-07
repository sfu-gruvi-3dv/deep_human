# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:39:00 2017
author: Kel
original author: Walid Benbihi
"""

# directories
test_dir        = 'data'
meanRgb_dir     = 'params/meanstd'

log_dir         = 'log_dir'
model_dir       = 'models/saved_depth_model/'
normal_dir = 'models/saved_normal_model/'

output_dir = 'output'
# Training Parameters
learning_rate 	= 5.0e-5 # Learning Rate
nEpochs 			= 60		# Number of epochs
iter_by_epoch 	= 5000	# Number of batch to train in one epoch
batch_size 		= 1		# Batch Size per iteration
step_to_save 	= 500   # number of steps to save model


# Hourglass Parameters
nStacks 		   = 2	 	# Number of stacks
outDim 		   = 304		# Number of output channels (number of joints or body parts)
nFeat 	      = 256		# Number of feature channels 
nLow 			   = 4 		# Number of downsampling


# Test parameters
num_steps      = 10     # Number of test images for calculating mean IOU
num_imgs       = 1     # Number of test images for visualization




