from functions.layers.downsampler import downsampler
import tensorflow as tf
import SimpleITK as sitk

import matplotlib.pyplot as plt
import numpy as np

def derivative_LoG( input):
    """

    :param input:
    :return:
    """

    kernelDimension = 3  # len(np.shape(input_numpy))


    kenelStrides = tuple([1] * kernelDimension)

    x_pad = tf.cast(input,tf.float64)
    LoGKernel = [[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]],
                         [[-1.,-1.,-1.],[-1.,26.,-1.],[-1.,-1.,-1.]],
                         [[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]]

    LoGKernel = np.expand_dims(LoGKernel, -1)
    LoGKernel = np.expand_dims(LoGKernel, -1)
    LoGKernel = tf.constant(LoGKernel)

    GoL = tf.concat([tf.nn.convolution(x_pad[:, :, :, :, i, tf.newaxis], LoGKernel, 'SAME', strides=kenelStrides)
                    for i in range(int(x_pad.get_shape()[4]))], axis=-1)

    return GoL


# loading image
# /srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/04_PP2_04_cinema2/arterial_sig_7.nii
# input=sitk.GetArrayFromImage(sitk.ReadImage('/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/test_data/1.mha'))
input=sitk.GetArrayFromImage(sitk.ReadImage('/exports/lkeb-hpc/syousefi/Data/Original_altas/BrainWeb/04/subject04_t1w_p4_brain.nii'))
# plt.figure()
# plt.show(input[50,:,:])
# plt.show()
img=tf.placeholder(tf.float32,shape=[1,150,150,150,1])

img0 = np.expand_dims(np.expand_dims(input[0:150,50:200,50:200],axis=0),axis=-1)
sess=tf.Session()

LoG=derivative_LoG( img)
result=sess.run(LoG,feed_dict={img:img0})
plt.figure()
R=result[0,40,:,:,0]
R[R>0]=0
plt.imshow(-R,cmap='gray')
plt.figure()
plt.imshow(img0[0,40,:,:,0],cmap='gray')
plt.show()
