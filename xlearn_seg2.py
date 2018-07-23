# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:07:45 2018

@author: Hannah
"""

import xlearn
from xlearn.segmentation import seg_train, seg_predict
import dxchange

batch_size= 800
nb_epoch= 50
nb_down= 3
nb_gpu= 4

opath='C:/Users/Hannah/Downloads/python programs/original_input/'
wpath=opath+'weights_seg.h5'

# create 2D arrays with single training image
imgx=dxchange.read_tiff(opath+'Dataset 2/249/249_original.tif')
imgy=dxchange.read_tiff(opath+'Dataset 2/249/249_bw_label.tif')


'''
# create 3D arrays with multiple training images
imgx=[]
imgy=[]
nums=['249', '252']
for item in nums:
    imgx.append(dxchange.read_tiff(opath+'original/'+item+'_original.tif'))
    imgy.append(dxchange.read_tiff(opath+'label/'+item+'_label.tif'))
'''

mdl=seg_train(imgx, imgy, batch_size=batch_size, nb_epoch=nb_epoch, nb_gpu=nb_gpu)
mdl.save_weights(wpath)

img_test=dxchange.read_tiff(opath+'original/252_original.tif')

seg_predict(img_test, wpath, opath, nb_down=nb_down, nb_gpu=nb_gpu)