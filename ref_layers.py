from collections import OrderedDict

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tcl
import tensorflow.contrib.slim as slim
# from IPython import embed
from tensorflow.contrib.framework import add_arg_scope
from tensorflow.contrib.layers import avg_pool2d, max_pool2d

from six.moves import xrange
# from tf_unet import util
# from tf_unet.layers import cross_entropy, pixel_wise_softmax_2
from tf_unet.layers import (bias_variable, conv3d, crop_and_concat,
                            deconv3d, weight_variable, weight_variable_devonc,max_pool)

SE_loss = tf.nn.sparse_softmax_cross_entropy_with_logits

# default_norm_K = 100.0
# default_norm_b = 80.0

default_norm_K = 1.0
default_norm_b = 0.0


def map_to_real_loss(x):
    return tf.real(x) + tf.imag(x)


def leaky_relu(x, alpha=0.2):
    return tf.maximum(tf.minimum(0.0, alpha * x), x)


def leaky_relu_batch_norm(x, alpha=0.2):
    return leaky_relu(tcl.batch_norm(x), alpha)


def relu(x):
    return tf.nn.relu(x)


def relu_batch_norm(x):
    return relu(tcl.batch_norm(x))


def dist_func(x, func_name):
    if func_name == "square":
        return tf.square(tf.abs(x))
    elif func_name == "abs":
        return tf.abs(x)
    else:
        raise ValueError("Unsupported dist_func type {}"
                         "".format(func_name))


def dual_channel_to_complex(x):
    # Change from two channel representation to complex type
    x_real = x[..., 0][..., None]
    x_imag = x[..., 1][..., None]
    return tf.complex(x_real, x_imag)


def fftshift_b01c(x, axes=(1, 2), inv=False):
    # fft shift function
    x_shifted = x
    for axis in axes:
        x_shifted = tf.concat(
            tf.split(x_shifted, 2, axis=axis)[::-1], axis=axis)

    return x_shifted


def fft3d_b01c(x_b01c):
    h = int(x_b01c.get_shape()[1])
    w = int(x_b01c.get_shape()[2])
    assert h == w
    # fft2d for b01c type images. fft2d performs only for inner most 2 dims, so
    # we need transpose to put w, h dim to the final 2 dims.
    x_bc01 = tf.transpose(x_b01c, (0, 4, 1, 2, 3))
    fft_x_bc01 = tf.fft3d(x_bc01)
    #fft_x_b01c = tf.transpose(fft_x_bc01, (0, 2, 3, 4, 1))
    return fftshift_b01c(fft_x_b01c) / w


def ifft3d_b01c(x_b01c):
    h = int(x_b01c.get_shape()[1])
    w = int(x_b01c.get_shape()[2])
    assert h == w
    # fft2d for b01c type images. fft2d performs only for inner most 2 dims, so
    # we need transpose to put w, h dim to the final 2 dims.
    x_bc01 = tf.transpose(fftshift_b01c(x_b01c), (0, 4, 1, 2,3))
    ifft_x_bc01 = tf.ifft2d(x_bc01)
    ifft_x_b01c = tf.transpose(ifft_x_bc01, (0,2, 3,4, 1))
    return ifft_x_b01c * w


def ifft_image_summary(x, name, max_image_summary, clip_val):
    ifft_x = tf.clip_by_value(
        abs(ifft2d_b01c(x[:max_image_summary])),
        -clip_val, clip_val)
    return tf.summary.image(name, ifft_x, max_outputs=max_image_summary)


def ifft_histogram_summary(x, name, clip_val):
    ifft_x = tf.clip_by_value(
        abs(ifft2d_b01c(x)),
        -clip_val, clip_val)
    return tf.summary.histogram(name, ifft_x)


def int_shape(x):
    return list(map(int, x.get_shape()[1:]))


def normalize(layer, K=default_norm_K, b=default_norm_b):
    # return layer / 127.5 - 1.
    return (layer - b) / K
    # return layer


def normalize_ksp(layer, ksp_weight, order=0):

    if order != 0:
        return layer * (ksp_weight)**(order)

    return layer


def denormalize_ksp(layer, ksp_weight, order=0):

    if order != 0:
        return layer * (ksp_weight)**(-order)

    return layer


def denormalize(layer, K=default_norm_K, b=default_norm_b):
    # return to exactly the original range
    return layer * K + b
    # Move to zero-one range
    # return (layer + 1.) / 2.
    # return (layer + 1.) / 2.
    # return layer


def _update_dict(layer_dict, scope, layer):
    name = "{}/{}".format(tf.get_variable_scope().name, scope)
    layer_dict[name] = layer


def image_from_paths(paths, shape, is_grayscale=True, seed=None):
    # Original code that loads images
    filename_queue = tf.train.string_input_producer(
        list(paths), shuffle=False, seed=seed)
    reader = tf.WholeFileReader()
    filename, data = reader.read(filename_queue)
    image = tf.image.decode_png(data, channels=3, dtype=tf.uint8)
    if is_grayscale:
        image = tf.image.rgb_to_grayscale(image)
    image.set_shape(shape)

    return filename, tf.to_float(image)


def kspace_from_paths(paths, key_image, key_m, seed=None):

    # Pre-load all hdf5 files (Yes, call me lazy)
    # We probably need a better solution later on
    kspace_img = []
    kspace_m = []
    for fname in paths:
        with h5py.File(fname, "r") as h5file:
            kspace_img += [h5file[key_image].value]
            kspace_m += [h5file[key_m].value]
    kspace_img = np.asarray(kspace_img)
    kspace_m = np.asarray(kspace_m)
    kspace_img = tf.Variable(kspace_img, name="kspace_img")
    kspace_m = tf.Variable(kspace_m, name="kspace_m")

    # Make as input queue
    kspace_img_queue = tf.train.input_producer(kspace_img, shuffle=False)
    kspace_m_queue = tf.train.input_producer(kspace_m, shuffle=False)

    return kspace_img_queue.dequeue(), kspace_m_queue.dequeue()


def real_conv2d(in_node, filter_size, features_in, features_out, activation_fn,
                pool_method, stride=None, pool_size=None, keep_prob=1.0,
                padding="CYCLIC"):

    if (stride is None) and (pool_size is None):
        if pool_method == "stride":
            stride = 2
            pool_size = 2
        else:
            stride = 1
            pool_size = 2

    with tf.variable_scope("conv"):
        w = weight_variable(
            [filter_size, filter_size,
             features_in, features_out], 0.02)
        b = bias_variable([features_out], 0.0)
    conv = conv3d(
        in_node, w, keep_prob, stride
    )
    # Bias
    in_node = conv + b

    # Activation
    if activation_fn is not None:
        in_node = activation_fn(in_node)

    # Pool
    if pool_method != "stride":
        if pool_method == "max":
            in_node = max_pool2d(
                in_node, pool_size, pool_size)
        elif pool_method == "avg":
            in_node = avg_pool2d(
                in_node, pool_size, pool_size)

    return in_node


def complex_conv3d(in_node_real, in_node_imag, filter_size, features_in,
                   features_out, activation_fn, pool_method, stride=None,
                   pool_size=None, keep_prob=1.0, padding="CYCLIC"):

    if (stride is None) and (pool_size is None):
        if pool_method == "stride":
            stride = 2
            pool_size = 2
        else:
            stride = 1
            pool_size = 2

    with tf.variable_scope("conv"):
        with tf.variable_scope("real"):
            w_real = weight_variable(
                [filter_size, filter_size,filter_size,
                 features_in, features_out], 0.02)
            b_real = bias_variable([features_out], 0.0)
        with tf.variable_scope("imag"):
            w_imag = weight_variable(
                [filter_size, filter_size, filter_size,
                 features_in, features_out], 0.02)
            b_imag = bias_variable([features_out], 0.0)
    conv_real = conv3d(
        in_node_real, w_real, keep_prob, stride
    ) - conv3d(
        in_node_imag, w_imag, keep_prob, stride
    )
    conv_imag = conv3d(
        in_node_real, w_imag, keep_prob, stride
    ) + conv3d(
        in_node_imag, w_real, keep_prob, stride
    )

    # Bias
    in_node_real = conv_real + b_real
    in_node_imag = conv_imag + b_imag

    # Activation
    if activation_fn is not None:
        with tf.variable_scope("real"):
            in_node_real = activation_fn(in_node_real)
        with tf.variable_scope("imag"):
            in_node_imag = activation_fn(in_node_imag)

    # Pool
    if pool_method != "stride":
            in_node_real = max_pool(
                in_node_real, pool_size)
            in_node_imag = max_pool(
                in_node_imag, pool_size)

    return in_node_real, in_node_imag


# def complex_conv2d(in_node_real, in_node_imag, filter_size, features_in,
#                    features_out, activation_fn, pool_method,
#                    stride=None, pool_size=None, keep_prob=1.0):

def complex_deconv3d(in_node_real, in_node_imag, filter_size, features_in,
                     features_out, activation_fn, stride, padding="CYCLIC"):

    with tf.variable_scope("deconv"):
        with tf.variable_scope("real"):
            wd_real = weight_variable_devonc(
                [filter_size, filter_size,filter_size,
                 features_out, features_in], 0.02)
            bd_real = bias_variable([features_out], 0.0)
        with tf.variable_scope("imag"):
            wd_imag = weight_variable_devonc(
                [filter_size, filter_size,filter_size,
                 features_out, features_in], 0.02)
            bd_imag = bias_variable([features_out], 0.0)

    o_size=in_node_real.get_shape()
    deconv_real = deconv3d(
        in_node_real, wd_real, stride, o_size,
    ) - deconv3d(
        in_node_imag, wd_imag, stride, o_size,
    ) + bd_real
    deconv_imag = deconv3d(
        in_node_real, wd_imag, stride, o_size,
    ) + deconv3d(
        in_node_imag, wd_real, stride, o_size,
    ) + bd_imag
    in_node_real = deconv_real
    in_node_imag = deconv_imag

    if activation_fn is not None:
        with tf.variable_scope("real"):
            in_node_real = activation_fn(in_node_real)
        with tf.variable_scope("imag"):
            in_node_imag = activation_fn(in_node_imag)

    return in_node_real, in_node_imag


# Conv Unet code
@add_arg_scope
def create_conv_net_complex(x, channels, n_class, layers=3,
                    features_root=16, filter_size=4, pool_size=2,
                    max_features=512, pool_method="stride", padding="CYCLIC"):
    """
    Creates a new convolutional unet for the given parametrization.

    :param x: input tensor, shape [?,nx,ny,channels] -- should be complex
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    if x.dtype != tf.complex64:
        raise ValueError(
            "The current implementation is for complex numbers!")

    summary_list = []

    # logging.info(
    #     "Layers {layers}, features {features}, filter size "
    #     "{filter_size}x{filter_size}, pool size: "
    #     "{pool_size}x{pool_size}".format(
    #         layers=layers,
    #         features=features_root,
    #         filter_size=filter_size,
    #         pool_size=pool_size))
    # Placeholder for the input image
    # embed()
    # nx = tf.shape(x)[1]
    # ny = tf.shape(x)[2]
    x_shp = [_s if _s is not None else -1 for _s in x.get_shape().as_list()]
    nx = x_shp[1]
    ny = x_shp[2]
    nz = x_shp[3]
    # x_image = tf.reshape(x, [-1, nx, ny, channels])
    x_image = tf.reshape(x, tf.stack([-1, nx, ny,nz, channels]))
    in_node_real = tf.real(x_image)
    in_node_imag = tf.imag(x_image)
    # batch_size = None
    # batch_size = x_image.get_shape().as_list()[0]

    if pool_method == "stride":
        stride = 2
    else:
        stride = 1

    # weights = []
    # biases = []
    # convs = []
    # pools = OrderedDict()
    # deconv = OrderedDict()
    dw_convs_real = OrderedDict()
    dw_convs_imag = OrderedDict()
    up_convs_real = OrderedDict()
    up_convs_imag = OrderedDict()
    #up_convs_2nd_real = OrderedDict()
    #up_convs_2nd_imag = OrderedDict()

    # Original input as -1 dw_h_conv layer
    dw_convs_real[-1] = in_node_real
    dw_convs_imag[-1] = in_node_imag

    in_size = 1000
    size = in_size

    num_features = [int(min(max_features, 2**layer * features_root))
                    for layer in xrange(layers)]
    # embed()
    # down layers
    print("Conv down---")
    with tf.variable_scope("convdown"):
        for layer in range(0, layers):
            with tf.variable_scope("level" + str(layer)):
                # Drop only when going down (except last layer)
                if layer < layers - 1:
                    # keep_prob = 0.5
                    keep_prob = 1.0  # no dropout
                else:
                    keep_prob = 1.0
                # No batch norm on first layer
                #if layer == 0:
                activation_fn = leaky_relu
                #else:
                #    activation_fn = leaky_relu_batch_norm
                features_out = num_features[layer]
                if layer > 0:
                    features_in = num_features[layer - 1]
                else:
                    features_in = channels
                # stddev = np.sqrt(2.0 / (filter_size ** 2 * features_out))
                stddev = 0.02
                print("initializing with std {:e}".format(stddev))
                # Conv with strides instead of pooling!
                in_node_real, in_node_imag = complex_conv3d(
                    in_node_real, in_node_imag, filter_size, features_in,
                    features_out, activation_fn, pool_method, padding=padding)

                dw_convs_real[layer] = in_node_real
                dw_convs_imag[layer] = in_node_imag

            # from IPython import embed
            # embed()
            print("output shape = {}".format(dw_convs_real[layer].get_shape()))

    in_node_real = dw_convs_real[layers - 1]
    in_node_imag = dw_convs_imag[layers - 1]
    # embed()
    # up layers
    # Note that uplayers do not perform any dropouts
    print("Conv up---")
    with tf.variable_scope("convup"):
        for layer in range(layers - 2, -2, -1):
            # print(layer)
            with tf.variable_scope("level" + str(layer)):
                # No dropout when up
                keep_prob = 1.0
                # Always relu with batch norm
                activation_fn = relu#relu_batch_norm
                features_in = num_features[layer + 1]

                # feautres_out
                if layer == -1:
                    features_out = n_class
                else:
                    features_out = num_features[layer]

                #stddev = np.sqrt(2.0 / (filter_size ** 2 * features_out))
                stddev = 0.02
                print("initializing with std {:e}".format(stddev))
                # Bring in skip connections
                if layer + 1 < layers - 1:
                    in_node_real = crop_and_concat(
                        dw_convs_real[layer + 1], in_node_real)
                    in_node_imag = crop_and_concat(
                        dw_convs_imag[layer + 1], in_node_imag)
                    features_in = features_in * 2

                print("output shape after crop concat= {}".format(in_node_real.get_shape()))
                # Unpool if necessary
                if pool_method != "stride":
                    with tf.variable_scope("unpool"):
                        in_node_real, in_node_imag = complex_deconv3d(
                            in_node_real, in_node_imag, pool_size,
                            features_in, features_in, None, pool_size,
                            padding=padding)

                # Perform deconv
                #print("in shape real = {}".format(in_node_real.get_shape()))
                #print("in shape imag = {}".format(in_node_imag.get_shape()))

                with tf.variable_scope("deconv"):
                    in_node_real, in_node_imag = complex_deconv3d(
                        in_node_real, in_node_imag, filter_size,
                        features_in, features_in, activation_fn, stride,
                        padding=padding)

                print("output shape after deconv= {}".format(in_node_real.get_shape()))
                up_convs_real[layer] = in_node_real
                up_convs_imag[layer] = in_node_imag

                with tf.variable_scope("conv"):
                    if layer != -1:
                        cur_activation = activation_fn
                    else:
                        cur_activation = None
                    in_node_real, in_node_imag = complex_conv3d(
                        in_node_real, in_node_imag, filter_size, features_in,
                        features_out, cur_activation, "stride", 1,
                        padding=padding)

                print("output shape = {}".format(in_node_real.get_shape()))

    # with tf.variable_scope("last_unet"):
    #     in_node_real, in_node_imag = complex_conv2d(
    #         in_node_real, in_node_imag, 1, features_out,
    #         n_class, None, None, None, None)
    # up_convs_real["out"] = in_node_real
    # up_convs_imag["out"] = in_node_imag

    output_map = tf.complex(in_node_real, in_node_imag)

    # output_map = activation_fn(conv + bias)
    # summary_list += [tf.summary.histogram("unetdebug/output-conv", conv)]
    # summary_list += [tf.summary.histogram("unetdebug/output-weight", weight)]
    # summary_list += [tf.summary.histogram("unetdebug/output-bias", bias)]
    # summary_list += [tf.summary.histogram("unetdebug/output-in_node", in_node)]
    # summary_list += [tf.summary.histogram(
    #     "unetdebug/output-output_map", output_map)]

    # if summaries:
    #     for i, (c1, c2) in enumerate(convs):
    #         tf.summary.image('summary_conv_%02d_01' % i,
    #                          get_image_summary(c1))
    #         tf.summary.image('summary_conv_%02d_02' % i,
    #                          get_image_summary(c2))

    #     for k in pools.keys():
    #         tf.summary.image('summary_pool_%02d' %
    #                          k, get_image_summary(pools[k]))

    #     for k in deconv.keys():
    #         tf.summary.image('summary_deconv_concat_%02d' %
    #                          k, get_image_summary(deconv[k]))
    # def get_image_summary(img, idx=0):
    #     """
    #     Make an image summary for 4d tensor image with index idx

    #     """

    #     V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    #     V -= tf.reduce_min(V)
    #     V /= tf.reduce_max(V)
    #     V *= 255

    #     img_w = tf.shape(img)[1]
    #     img_h = tf.shape(img)[2]
    #     V = tf.reshape(V, tf.stack((img_w, img_h, 1)))
    #     V = tf.transpose(V, (2, 0, 1))
    #     V = tf.reshape(V, tf.stack((-1, img_w, img_h, 1)))
    #     return V

    # for k in dw_convs_real.keys():
    #     summary_list += [
    #         tf.summary.histogram("dw_convolution_real_%02d" %
    #                              k + '/activations', dw_convs_real[k])
    #     ]
    # for k in dw_convs_imag.keys():
    #     summary_list += [
    #         tf.summary.histogram("dw_convolution_imag_%02d" %
    #                              k + '/activations', dw_convs_imag[k])
    #     ]

    # for k in deconv.keys():
    #     summary_list += [
    #         tf.summary.image('summary_deconv_concat_%02d' %
    #                          k, get_image_summary(deconv[k]))
    #     ]

    # for k in up_convs_real.keys():
    #     summary_list += [
    #         tf.summary.histogram("up_convolution_real_%s" %
    #                              k + '/activations', up_convs_real[k])
    #     ]
    # for k in up_convs_imag.keys():
    #     summary_list += [
    #         tf.summary.histogram("up_convolution_imag_%s" %
    #                              k + '/activations', up_convs_imag[k])
    #     ]

    return output_map, int(in_size - size), summary_list


# @add_arg_scope
# def resnet_block(
#         inputs, scope, num_outputs=64, kernel_size=[3, 3],
#         stride=[1, 1], padding="SAME", layer_dict={}):
#     with tf.variable_scope(scope):
#         layer = conv2d(
#             inputs, num_outputs, kernel_size, stride,
#             padding=padding, activation_fn=tf.nn.relu, scope="conv1")
#         layer = conv2d(
#             inputs, num_outputs, kernel_size, stride,
#             padding=padding, activation_fn=None, scope="conv2")
#         outputs = tf.nn.relu(tf.add(inputs, layer))
#     _update_dict(layer_dict, scope, outputs)
#     return outputs


# @add_arg_scope
# def repeat(inputs, repetitions, layer, layer_dict={}, **kargv):
#     outputs = slim.repeat(inputs, repetitions, layer, **kargv)
#     _update_dict(layer_dict, kargv['scope'], outputs)
#     return outputs


# @add_arg_scope
# def conv2d(inputs, num_outputs, kernel_size, stride,
#            layer_dict={}, activation_fn=tf.nn.relu,
#            # weights_initializer=tf.random_normal_initializer(0, 0.001),
#            # weights_initializer=tf.contrib.layers.xavier_initializer(),
#            # biases_initializer=tf.contrib.layers.xavier_initializer(),
#            weights_initializer=tf.truncated_normal_initializer(stddev=0.02),
#            # biases_initializer=tf.truncated_normal_initializer(stddev=0.02),
#            biases_initializer=tf.constant_initializer(0.0),
#            scope=None, name="", **kargv):
#     outputs = slim.conv2d(
#         inputs, num_outputs, kernel_size,
#         stride, activation_fn=activation_fn,
#         weights_initializer=weights_initializer,
#         # KMYI:170117
#         # Fix for TF version
#         # biases_initializer=tf.constant_initializer(0.0),
#         biases_initializer=biases_initializer,
#         scope=scope, **kargv)
#     if name:
#         scope = "{}/{}".format(name, scope)
#     _update_dict(layer_dict, scope, outputs)
#     return outputs


# @add_arg_scope
# def max_pool2d(inputs, kernel_size=[3, 3], stride=[1, 1],
#                layer_dict={}, scope=None, name="", **kargv):
#     outputs = slim.max_pool2d(inputs, kernel_size, stride, **kargv)
#     if name:
#         scope = "{}/{}".format(name, scope)
#     _update_dict(layer_dict, scope, outputs)
#     return outputs


# @add_arg_scope
# def tanh(inputs, layer_dict={}, name=None, **kargv):
#     outputs = tf.nn.tanh(inputs, name=name, **kargv)
#     _update_dict(layer_dict, name, outputs)
#     return outputs
