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
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np


def rsnr(rec,oracle):
    sumP    =        sum(oracle.reshape(-1))
    sumI    =        sum(rec.reshape(-1))
    sumIP   =        sum( oracle.reshape(-1) * rec.reshape(-1) )
    sumI2   =        sum(rec.reshape(-1)**2)
    A       =        np.matrix([[sumI2, sumI],[sumI, oracle.size]])
    b       =        np.matrix([[sumIP],[sumP]])
    c       =        np.linalg.inv(A)*b
    rec     =        c[0,0]*rec+c[1,0]
    err     =        sum((oracle.reshape(-1)-rec.reshape(-1))**2)
    SNR     =        10.0*np.log10(sum(oracle.reshape(-1)**2)/err)
    return SNR

def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def _denorm_data(data):

        Nx=data.shape[1]
        Ny=data.shape[2]
        Nz=data.shape[3]
        xx = np.matmul(np.arange(Nx).reshape(
                (-1, 1)), np.ones((1, Ny))) - Nx / 2.0
        xx = np.matmul(xx.reshape(Nx,Ny,1), np.ones((1,1,Nz)))

        yy = np.matmul(np.ones((Nx,1)),np.arange(Ny).reshape(1, Ny)) - Ny / 2.0
        yy = np.matmul(yy.reshape(Nx,Ny,1), np.ones((1,1,Nz)))
        zz = np.ones((Nx,Ny,1))
        zz = np.matmul(zz, np.arange(Nz).reshape(1,1,Nz)  ) - Nz/2.0

        ksp_w = np.maximum(1,(xx**2 + yy**2+zz**2)**0.5)

        data /= ksp_w[...,None]
        return data




def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)

def relu(x):
    return tf.nn.relu(x)

def fft3d_b01c(x_b01c):
    #x_shape=tf.to_int64(tf.shape(x_b01c))
    x = int(x_b01c.get_shape()[1])
    y = int(x_b01c.get_shape()[2])
    z = int(x_b01c.get_shape()[3])
    # fft2d for b01c type images. fft2d performs only for inner most 2 dims, so
    # we need transpose to put w, h dim to the final 2 dims.
    x_bc01 = tf.transpose(x_b01c, (0, 4, 1, 2, 3))

    fft_x_bc01 = tf.fft3d(x_bc01)
    fft_x_b01c = tf.transpose(fft_x_bc01, (0, 2, 3, 4, 1))
    return (fft_x_b01c) /(x*y*z)

def ifft3d_b01c(x_b01c):
    #x_shape=tf.shape(x_b01c)
    x = int(x_b01c.get_shape()[1])
    y = int(x_b01c.get_shape()[2])
    z = int(x_b01c.get_shape()[3])

    # fft2d for b01c type images. fft2d performs only for inner most 2 dims, so
    # we need transpose to put w, h dim to the final 2 dims.
    x_bc01 = tf.transpose(fftshift_b01c(x_b01c), (0, 4, 1, 2, 3))

    ifft_x_bc01 = tf.ifft3d(x_bc01)
    ifft_x_b01c = tf.transpose(ifft_x_bc01, (0, 2, 3, 4, 1))
    return ifft_x_b01c *x*y*z


def fftshift_b01c(x, axes=(1,2,3), inv=False):
    # fft shift function
    x_shifted = x
    for axis in axes:
        x_shifted = tf.concat(
            tf.split(x_shifted, 2, axis=axis)[::-1], axis=axis)

    return x_shifted

def ifft_image_summary(x, name, max_image_summary):
    ifft_x = abs(ifft3d_b01c(x[:max_image_summary]))
    return tf.summary.image(name, ifft_x, max_outputs=max_image_summary)


def normalize_ksp(layer, ksp_weight, order=0):

    if order != 0:
        return layer * (ksp_weight)**(order)

    return layer


def denormalize_ksp(layer, ksp_weight, order=0):

    if order != 0:
        return layer * (ksp_weight)**(-order)

    return layer

def relu_batch_norm(x):
    return relu(tcl.batch_norm(x))

def batch_norm_nd(x):
    return tf.contrib.layers.batch_norm(x)

#################################################################
#def weight_variable(shape, stddev=0.1):
#    initial = tf.truncated_normal(shape, stddev=stddev)
#    return tf.Variable(initial)

#def weight_variable_devonc(shape, stddev=0.1):
#    return tf.Variable(tf.truncated_normal(shape, stddev=stddev))

#def bias_variable(shape):
#    initial = tf.constant(0.1, shape=shape)
#    return tf.Variable(initial)

#################################################################
def weight_variable(shape, stddev=0.02, name="weight"):
    initial = tf.random_normal_initializer(stddev=stddev)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def weight_variable_devonc(shape, stddev=0.02, name="weight"):
    initial = tf.random_normal_initializer(stddev=stddev)
    return tf.get_variable(name=name, shape=shape, initializer=initial)


def bias_variable(shape, stddev=0.1, name="bias"):
    if stddev > 0:
        initial = tf.random_normal_initializer(stddev=stddev)
    else:
        initial = tf.constant_initializer(0.0)
    return tf.get_variable(name=name, shape=shape, initializer=initial)

#################################################################
def conv3d(x, W,keep_prob_,stride):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, stride, stride, stride, 1], padding='SAME')
    return tf.nn.dropout(conv_3d, keep_prob_)

def deconv3d(x, W,stride,x_size,batchN):
    #x_size = [_s if _s is not None else -1 for _s in x.get_shape().as_list()]
    #x_size=tf.shape(x)
    dyn_input_shape = tf.shape(x)
    batch_size = dyn_input_shape[0]
    w_shape = [_s if _s is not None else -1 for _s in W.get_shape().as_list()]

    output_shape = tf.stack([batchN, x_size[1]*stride, x_size[2]*stride, x_size[3]*stride, w_shape[3]])
    res= tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride,stride, 1], padding='SAME')
    return res
    #return tf.reshape(res,output_shape)

    #return tf.nn.conv3d_transpose(x, W, output_shape, strides=[1, stride, stride,stride, 1], padding='SAME')

def max_pool(x,n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='SAME')

#def crop_and_concat(x1,x2):
#    x1_shape = tf.shape(x1)
#    x2_shape = tf.shape(x2)
#    # offsets for the top left corner of the crop
#    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2, 0]
#    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
#    x1_crop = tf.slice(x1, offsets, size)
#    return tf.concat(4, [x1_crop, x2])

def crop_and_concat(x1, x2):
    # x1_shape = tf.shape(x1)
    # x2_shape = tf.shape(x2)
    x1_shape = [_s if _s is not None else -
                1 for _s in x1.get_shape().as_list()]
    x2_shape = [_s if _s is not None else -
                1 for _s in x2.get_shape().as_list()]
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
               (x1_shape[2] - x2_shape[2]) // 2,(x1_shape[3] - x2_shape[3]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size)
    # return tf.concat(3, [x1_crop, x2])
    return tf.concat([x1_crop, x2], axis=4)

def pixel_wise_softmax(output_map):
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map,tf.reverse(exponential_map,[False,False,False,True]))
    return tf.div(exponential_map,evidence, name="pixel_wise_softmax")

def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1,1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)

def euclidean_loss (y_,output_map):
	return tf.nn.l2_loss(tf.abs(y_-output_map))

def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")
#     return tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output_map), reduction_indices=[1]))
