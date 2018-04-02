# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.

'''
Created on Jul 28, 2016

author: jakeret
forked for 3D BPConvNet
modified at Feb 2018
modified by : Kyong Jin (kyonghwan.jin@gmail.com)

# input data dimension : NCXYZ ( batch x channels x X x Y x Z )

running command example
python main.py --lr=1e-4 --output_path='logs/' --features_root=32 --layers=5 --restore=False
'''
import tensorflow as tf
from tf_unet import unet, util, image_util,layers
import h5py
import numpy as np
import os


flags=tf.app.flags
flags.DEFINE_integer('features_root',32,'learning rate')
flags.DEFINE_integer('layers',5,'number of depth')
flags.DEFINE_integer('channels',8,'number of channels in data')
flags.DEFINE_string('optimizer','adam','optimizing algorithm : adam / momentum')
flags.DEFINE_float('lr',1e-4,'learning rate')
flags.DEFINE_boolean('restore',True,'resotre model')
flags.DEFINE_string('output_path','logs/unet3d/wdec/','log folder')
flags.DEFINE_string('data_path','/Volumes/Disk_addition/mr_data/mr_data/3d_wdec/*.h5','log folder')
flags.DEFINE_string('test_path','/Volumes/Disk_addition/mr_data/mr_data/3d_wdec/test/*.h5','log folder')
flags.DEFINE_boolean('is_training',True,'training phase/ deploying phase')
conf=flags.FLAGS

if __name__ =='__main__':

    data_provider=image_util.ImageDataProvider_hdf5_vol(conf.data_path,conf.channels)
    data_provider_test=image_util.ImageDataProvider_hdf5_vol(conf.test_path,conf.channels,test=True)

    net=unet.Unet(layers=conf.layers,size_mask=[256//2,320//2,320//2],features_root=conf.features_root,channels=conf.channels, n_class=conf.channels,filter_size=3, cost='psnr')
    if conf.is_training:
        #setup & trainig
        if conf.optimizer=='adam':
            trainer =unet.Trainer(net,batch_size=1,optimizer="adam",opt_kwargs=dict(beta1=0.9,learning_rate=conf.lr))
        elif conf.optimizer=='momentum':
            trainer =unet.Trainer(net,batch_size=1,optimizer="momentum",opt_kwargs=dict(momentum=0.99,learning_rate=conf.lr))

        path=trainer.train(data_provider,conf.output_path,training_iters=17,epochs=1000,dropout=1,restore=True)

    else:
        save_path = os.path.join(conf.output_path, "model.cpkt")
        x_test, y_test = data_provider_test(1)
        prediction = net.predict(save_path, x_test)
        print("Testing avg RSNR: {:.2f}".format(layers.rsnr(prediction, y_test)))
