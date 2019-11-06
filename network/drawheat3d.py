#print('draw3d---------------------------------------------------')
import numpy as np
import math
from utils import util
import matplotlib.pyplot as plt
import os

out = np.zeros([64, 64, 16 * 19])
sigma_2d = 2
size_z = 2 * math.floor((6* sigma_2d * 19 / 64 +1) / 2) + 1
j3d = np.load('/home/sicong/output/joints.npy')[0]
for j in range(16):
    #if joints2d[1,j] >= img_full.shape[0] or joints2d[0,j] >=img_full.shape[1] or joints2d[1,j]<0 or joints2d[0,j]<0:
        #continue
    z = (int)(j3d[j,2])
    pt = j3d[j,:2].astype(np.int64)
    out[:,:,j * 19 : (j+1) * 19] = util.Drawguassian3D(out[:,:,j * 19 : (j+1) * 19], pt, z , sigma_2d, size_z)
    for k in range(19):
        plt.imsave(os.path.join('/home/sicong/output', '{}_{}_joint.png'.format(j,k)), out[:,:,j*19+k])