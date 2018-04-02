from tf_unet import unet, util, image_util, layers
import h5py
import numpy as np
import tensorflow as tf
from tf_unet.layers import _denorm_data
# data path
d_path='/Volumes/Disk_addition/mr_data/mr_data/3d_ksp/'
# log path
l_path='/Volumes/Disk_addition/logs/unet3d/'

dsr_grid=[4, 2, 2]
grid_id=[0,0,0]
h5f = h5py.File(d_path+'mask/mask3d_x10.h5', 'r')
mask=np.array(h5f['mask'])

data_provider=image_util.ImageDataProvider_hdf5_low_sampling(d_path+'*.h5',mask,dsr_grid=dsr_grid,num_patch=8,size_patch=16)


y_patch,cts,mask,xx,yy,mask=data_provider(1)
np.amax(xx)

#yy=_denorm_data(xx)
#yy_d=_denorm_data(yy)
#yy=yy_d
ysp=np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(yy,axes=(1,2,3)),axes=(1,2,3)),axes=(1,2,3))
np.amax(abs(ysp))

import matplotlib.pyplot as plt
plt.subplot(231)
plt.imshow(np.log(np.abs(np.squeeze(yy[0,yy.shape[1]//2,:,:,0]))), cmap='gray')
plt.subplot(232)
plt.imshow(np.log(np.abs(np.squeeze(yy[0,:,yy.shape[2]//2,:,0]))), cmap='gray')
plt.subplot(233)
plt.imshow(np.log(np.abs(np.squeeze(yy[0,:,:,yy.shape[3]//2,0]))), cmap='gray')
plt.subplot(234)
plt.imshow(np.sqrt(np.sum(np.squeeze(abs(ysp[0,ysp.shape[1]//2,:,:,:])**2),axis=2)), cmap='gray')
plt.subplot(235)
plt.imshow(np.sqrt(np.sum(np.squeeze(abs(ysp[0,:,ysp.shape[2]//2,:,:])**2),axis=2)), cmap='gray')
plt.subplot(236)
plt.imshow(np.sqrt(np.sum(np.squeeze(abs(ysp[0,:,:,ysp.shape[3]//2,:])**2),axis=2)), cmap='gray')
plt.show()

