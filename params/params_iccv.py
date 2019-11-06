# -*- coding: utf-8 -*-
"""
Created on Sun May 28 17:39:00 2017
author: Kel
original author: Walid Benbihi
"""

# directories
bg_dir = '/local-scratch2/normal_dataset/bg_dataset'

train_dir       = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/train'
valid_dir       = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/valid'
test_dir        = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/surreal/data/SURREAL/data/cmu/test'

real_train_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/real_data'
real_valid_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/real_data'
real_test_dir = '/media/sicong/a86d93af-1a2e-469b-972c-f819c47cd5ee/real_data'


# train_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/train_data'
# valid_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/train_data'
# test_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/test_data'

# train_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/train'
# valid_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/val'
# test_dir = '/media/vision-gpu/OLD/humen_depth_data/surreal/data/SURREAL/data/cmu/test'

# train_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/all_data'
# valid_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/all_data'
# test_dir = '/media/vision-gpu/OLD/humen_depth_data/detail_data/all_data'

meanRgb_dir     = '/home/sicong/deep_depth/meanstd'



# log_dir         = 'log_dir'
# model_dir       = 'saved_model/'

# log_dir         = 'log_dir_1'
# model_dir       = 'saved_model_1/'

# log_dir         = 'log_dir_2'
# model_dir       = 'saved_model_2/'

# log_dir         = 'log_dir_3'
# model_dir       = 'saved_model_3/'

# log_dir         = 'log_dir_pool'
# model_dir       = 'best_pool_model/'

# log_dir         = 'log_dir_4'
# model_dir       = 'saved_model_4/'

# log_dir         = 'log_dir_5'
# model_dir       = 'saved_model_5/'

# log_dir         = 'log_dir_6'
# model_dir       = 'saved_model_6/'

# log_dir         = 'log_dir_align'
# model_dir       = 'saved_model_align/'

# log_dir         = 'log_dir_ori'
# model_dir       = 'saved_model_ori/'

# log_dir         = 'log_dir_oriori'
# model_dir       = 'saved_model_oriori/'

# log_dir         = 'log_dir_bila'
# model_dir       = 'saved_model_bila/'

# log_dir         = 'log_dir_stage12'
# model_dir       = 'saved_model_stage12/'

log_dir         = 'log_dir_base'
model_dir       = '/home/sicong/deep_depth/models/saved_model_iccv_all/'

# log_dir         = 'log_dir_stage12'
# model_dir       = 'saved_model_old/'

# model_dir   = 'saved_model_lin_no/'
# model_dir   = 'saved_model_lin_with/'

depth_model_dir = '/media/sicong/DATA/saved_model_iccv_depth/'
normal_dir = '/home/sicong/Human_depth_multiview/normal_train/saved_normal_model/'
mask_dir = 'saved_model_mask/'
joint3d_model_dir = 'saved_3dpose_model/'
ab_model_dir = 'saved_ab_model/'
joint_refine_model_dir = 'saved_jointrefine_model/'
seg2d_dir = 'saved_seg2d_model/'
normal_dir = 'saved_normal_model/'
poseseg_model_dir = '/media/sicong/DATA/saved_model_base/'

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




