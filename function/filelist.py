import glob2
import pickle
import time

start_time = time.time()
#filelist = glob2.glob('/media/sicong/137AE4175F4F2DEC/data/surreal/data/SURREAL/data/cmu/train' + '/**/*.mp4')

filelist = glob2.glob('/media/sicong/137AE4175F4F2DEC/dataset_all' + '/**/*_rgb.png')

with open('train_list_detail.pkl', 'wb') as fout:
    pickle.dump(filelist, fout)
