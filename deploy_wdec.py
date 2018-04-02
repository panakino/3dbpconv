from tf_unet import unet, util, image_util
from tf_unet.layers import _denorm_data
import h5py
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
from calcul import psnr_loss_np as psnr_loss

dsr_grid=[1, 1, 1]
cmode="abs_sp" # "abs_sp","comp_sp","comp_ksp","real_sp"
size_patch=1
num_patch=1
channels=8
# data path
d_path='/Volumes/Disk_addition/mr_data/mr_data/'
# log path
l_path='/Volumes/Disk_addition/logs/unet3d/wdec/'+cmode+'/'

sname=l_path+'dsr'+repr(dsr_grid[0])+repr(dsr_grid[1])+repr(dsr_grid[2])

h5f = h5py.File(d_path+'3d_ksp/mask/mask3d_x6.h5', 'r')
mask=np.array(h5f['mask'])

data_provider=image_util.ImageDataProvider_hdf5_low_sampling_sp(d_path+'3d_wdec/test/*.h5',mask,dsr_grid,cmode,channels,test=True)

#setup & trainig
net=unet.Unet(layers=5,size_mask=[256//2,320//2,320//2],dsr_grid=dsr_grid,features_root=32,channels=8, n_class=8,filter_size=3,num_patch=num_patch, size_patch=size_patch, cost='psnr',mode=cmode)

path=sname+'/model.cpkt'
sess=tf.Session()


#xx,yy,_,_,_,_,_=data_provider(0)
xx=np.empty((1,128,160,160,channels))
y_dummy=np.empty((1,128,160,160,channels))
recon=net.predict_w_sess(path,xx,y_dummy,sess,True)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
for i in range(0,3):

    y_dummy=np.empty((1,128,160,160,channels))
    xx,yy,_,_,_,_,_=data_provider(i)
    a=time.time()
    #recon=net.predict(path,xx)
    #if i==0:
    #    opt=True
    #else:
    #    opt=False

    recon=net.predict_w_sess(path,xx,y_dummy,sess,False)
    b=time.time()
    print('time')
    print(b-a)
    #fig=plt.figure()
    rec_sp=recon

    gt_sp=yy
    down_sp=xx

    #down_sl=np.sqrt(np.sum(abs(np.squeeze(down_sp[:,:,xx.shape[2]//2,:,:]))**2,axis=2))
    #rec_sl=np.sqrt(np.sum(abs(np.squeeze(rec_sp[:,:,xx.shape[2]//2,:,:]))**2,axis=2))
    #gt_sl=np.sqrt(np.sum(abs(np.squeeze(gt_sp[:,:,xx.shape[2]//2,:,:]))**2,axis=2))

    psnr_down=psnr_loss(down_sp,gt_sp)
    psnr_rec=psnr_loss(rec_sp,gt_sp)
    print("psnr down",psnr_down)
    print("psnr recon",psnr_rec)

    #plt.subplot(131)
    #plt.imshow((down_sl), cmap='gray')
    #plt.title('down'+repr(psnr_down))
    #plt.subplot(132)
    #plt.imshow((rec_sl), cmap='gray')
    #plt.title('recon'+repr(psnr_rec))
    #plt.subplot(133)
    #plt.imshow((gt_sl),cmap='gray')
    #plt.title('ground-truth')
    #plt.show()

    #sav_path='/Volumes/Disk_addition/mr_data/mr_data/3d_wdec_res/'

    #f=h5py.File(sav_path+'res_P'+repr(i+1)+'.h5','w')
    #f.create_dataset('down',(8,128,160,160),data=np.float32(np.squeeze(xx[0,:,:,:,:])).transpose(3,0,1,2),dtype='float32')
    #f.create_dataset('rec',(8,128,160,160),data=np.float32(np.squeeze(recon[0,:,:,:,:])).transpose(3,0,1,2),dtype='float32')
    #f.create_dataset('ref',(8,128,160,160),data=np.float32(np.squeeze(yy[0,:,:,:,:])).transpose(3,0,1,2),dtype='float32')


