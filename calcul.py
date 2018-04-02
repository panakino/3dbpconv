import numpy as np
import tensorflow as tf
def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.log(tf.constant(10, dtype=numerator.dtype)))
    return numerator / denominator

def psnr_loss(refined_img, gt_img):
    psnr = 20. * log10(tf.reduce_max(abs(gt_img))) - 10.*log10(tf.reduce_mean(tf.square(abs(gt_img - refined_img))))
    return psnr

def log_mse(refined_img, gt_img):
    psnr = - 2.*log10(tf.reduce_mean(tf.square(abs(gt_img - refined_img))))
    return psnr


def log10_np(x):
    numerator = np.log(x)
    denominator = np.log(10.0)
    return numerator / denominator

def psnr_loss_np(refined_img, gt_img):
    psnr = 20. * log10_np(np.amax(abs(gt_img))) - 10.*log10_np(np.average(np.square(abs(gt_img - refined_img))))
    return psnr
