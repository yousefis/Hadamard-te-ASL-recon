from functions.loss.ssim_loss import *
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np


def seg_sig( segmentation, logits):
    one = tf.constant(1, shape=segmentation.shape,dtype=tf.float32)
    two = tf.constant(2, shape=segmentation.shape,dtype=tf.float32)
    three = tf.constant(3, shape=segmentation.shape,dtype=tf.float32)

    wm_mask = tf.cast(tf.equal(segmentation,three ),tf.float32)

    gm_mask = tf.cast(tf.equal(segmentation, two),tf.float32)

    csf_mask = tf.cast(tf.equal(segmentation,one ),tf.float32)

    WM = tf.multiply(logits, wm_mask)
    GM = tf.multiply(logits, gm_mask)
    CSF = tf.multiply(logits, csf_mask)

    return tf.reduce_mean(WM),tf.reduce_mean(GM),tf.reduce_mean(CSF)

if __name__=='__main__':
    path = '/exports/lkeb-hpc/syousefi/Data/Simulated_data/BrainWeb_permutation7/04_PP4_04_cinema1/decoded_crush/'
    seg='/exports/lkeb-hpc/syousefi/Data/Original_altas/BrainWeb/04/subject04_t1w_p4_brain_seg.nii'
    I1 = np.float32(sitk.GetArrayFromImage(sitk.ReadImage(path + 'crush_0.nii')))
    I2 = np.float32(sitk.GetArrayFromImage(sitk.ReadImage(seg)))

    X = tf.placeholder(tf.float32, (181, 256, 256))
    Y = tf.placeholder(tf.float32, (181, 256, 256))

    [WM,GM,CSF, wm_mask]=seg_ssim(segmentation=X, logits=Y)


    # mssim, ssim_map=SSIM(X, Y, max_val=1.0)



    sess=tf.Session()
    WM_, GM_, CSF_= sess.run([WM,GM,CSF],
             feed_dict={X:I2,Y:I1})

    plt.imshow(WM_[120, :, :])



    print('c')