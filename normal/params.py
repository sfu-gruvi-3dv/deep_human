# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:39:00 2017
author: Kel
original author: Walid Benbihi
"""

# directories
# train_dir       = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
# valid_dir       = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/valid'
# test_dir        = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/test'

# train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
# valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
# test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/datasets'
# meanRgb_dir     = '/home/sicong/surreal/meanRgb.t7'

# train_dir = '/local-scratch2/normal_dataset/textureless_deformable_surfaces'
# valid_dir = '/local-scratch2/normal_dataset/textureless_deformable_surfaces'
# test_dir = '/local-scratch2/normal_dataset/textureless_deformable_surfaces'
bg_dir = '/local-scratch2/normal_dataset/bg_dataset'
meanRgb_dir = '/local-scratch2/deep_volume/newversion/meanRgb.t7'
real_data_dir = '/local-scratch2/normal_dataset/real_data_2'
train_dir = real_data_dir
valid_dir = real_data_dir
test_dir = real_data_dir

log_dir         = 'log_dir'
model_dir       = 'saved_model/'
seg_results_dir = 'test_pose_2d_results/'
real_imgs_dir   = 'test_real_imgs/'
# Training Parameters
learning_rate 	= 2.5e-4 # Learning Rate 
nEpochs 			= 20		# Number of epochs
iter_by_epoch 	= 20000	# Number of batch to train in one epoch
batch_size 		= 2		# Batch Size per iteration
step_to_save 	= 1000	   # number of steps to save model


# Hourglass Parameters
nStacks 		   = 1	 	# Number of stacks
outDim 		   = 3		# Number of output channels (number of joints or body parts)
nFeat 	      = 256		# Number of feature channels 
nLow 			   = 4 		# Number of downsampling


# Test parameters
num_steps      = 10     # Number of test images for calculating mean IOU
num_imgs       = 1     # Number of test images for visualization

# Camera Intrinsics
fx = 1066.01 / 2.22
fy = 1068.87 / 2.22
cx = 945 / 2.22 -100
cy = 520 / 2.22


