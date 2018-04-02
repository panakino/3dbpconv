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
author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

#import cv2
import glob
import numpy as np
import h5py
from PIL import Image
import tensorflow as tf
from tf_unet.layers import (fft3d_b01c,ifft3d_b01c,fftshift_b01c)

class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """

    channels = 1
    n_class = 1


    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
        self.x=0
        self.y=0
        self.nb=0

    def _load_data_and_label(self):
        data,label= self._next_data()

        return data, label

        #real, imag, real_L, imag_L  = self._next_data()

        #real = self._process_data(real)
        #imag = self._process_data(imag)
        #real_L = self._process_data(real_L)
        #imag_L = self._process_data(imag_L)

        #train_data, labels = self._post_process(train_data, labels)

        #self.nb=nb
        # return (nc,nx,ny,nz) --> (nb,nx,ny,nz,nc)
            #return real.transpose(1,2,3,0).reshape(1,nx,ny,nz,nc), imag.transpose(1,2,3,0).reshape(1,nx,ny,nz,nc)
        #return real, imag, real_L, imag_L

    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels

        return label


    def __call__(self, n):
        data,label= self._load_data_and_label()
        nc=data.shape[0]
        nxh=data.shape[1]
        nyh=data.shape[2]
        nzh=data.shape[3]

        Xsp=np.float32(data.transpose(1,2,3,0).reshape(1,nxh,nyh,nzh,nc))
        Ysp=np.float32(label.transpose(1,2,3,0).reshape(1,nxh,nyh,nzh,nc))

        if self.test==False:
            if np.random.rand(1)>0.5:
                Xsp=Xsp[:,::-1,:,:,:]
                Ysp=Ysp[:,::-1,:,:,:]
            if np.random.rand(1)>0.5:
                Xsp=Xsp[:,:,::-1,:,:]
                Ysp=Ysp[:,:,::-1,:,:]
            if np.random.rand(1)>0.5:
                Xsp=Xsp[:,:,:,::-1,:]
                Ysp=Ysp[:,:,:,::-1,:]

        return Xsp,Ysp

class ImageDataProvider(BaseDataProvider):
    """
    Generic data provider for images, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = ImageDataProvider("..fishes/train/*.tif")

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'

    """

    n_class = 2

    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif'):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1

        self.data_files = self._find_data_files(search_path)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        img = self._load_file(self.data_files[0])
        #self.channels = 1 if len(img.shape) == 2 else img.shape[-1]

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if not self.mask_suffix in name]


    def _load_file(self, path, dtype=np.float32):
        return np.array(Image.open(path), dtype)
        # return np.squeeze(cv2.imread(image_name, cv2.IMREAD_GRAYSCALE))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = -1

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.file_idx]
        label_name = image_name.replace(self.data_suffix, self.mask_suffix)

        img = self._load_file(image_name, np.float32)
        label = self._load_file(label_name, np.bool)

        return img,label

class ImageDataProvider_hdf5(BaseDataProvider):
    """
    Generic data provider for images in hdf5 format, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix

    Usage:
    data_provider = ImageDataProvider_hdf5("..fishes/train/*.h5",batchsize)

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    n_class = 1

    def __init__(self, search_path):
        self.file_idx = -1
        self.data_files = self._find_data_files(search_path)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        img = self._load_file(self.data_files[0],'real')
        self.ids=np.random.permutation(len(self.data_files))

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return all_files

    def _load_file(self, path, opt):
        h5f = h5py.File(path, 'r')
        x = np.array(h5f[opt])
        return x

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0
            self.ids=np.random.permutation(len(self.data_files))

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.ids[self.file_idx]]

        img = self._load_file(image_name, 'real')
        label = self._load_file(image_name, 'imag')

        return img,label


class ImageDataProvider_hdf5_vol(BaseDataProvider):
    """
    Generic data provider for images in hdf5 format, supports gray scale and colored images.
    Assumes that the data images and label images are stored in the same folder
    and that the labels have a different file suffix

    Usage:
    data_provider = ImageDataProvider_hdf5("..fishes/train/*.h5",batchsize)

    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    n_class = 1

    def __init__(self, search_path,channels=8,test=False):
        self.file_idx = -1
        self.test=test
        self.channels=channels
        self.data_files = self._find_data_files(search_path)

        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))

        if self.test:
            self.ids=range(0,len(self.data_files))
        else:
            self.ids=np.random.permutation(len(self.data_files))

    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return all_files

    def _load_file(self, path, opt):
        h5f = h5py.File(path, 'r')
        x = np.array(h5f[opt])

	return x

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = -1
            self.ids=np.random.permutation(len(self.data_files))

    def _next_data(self):
        self._cylce_file()
        image_name = self.data_files[self.ids[self.file_idx]]
        #print("image_name",self.ids[self.file_idx])
        self.image_name=self.ids[self.file_idx]

        data= self._load_file(image_name, 'data')
        label = self._load_file(image_name, 'label')
        return data,label
