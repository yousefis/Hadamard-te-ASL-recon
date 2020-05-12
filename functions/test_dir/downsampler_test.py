from functions.layers.downsampler import downsampler
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from functions.layers.upsampling import upsampling
ds=downsampler()
input=sitk.GetArrayFromImage(sitk.ReadImage('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/04_PP2_04_cinema2/arterial_sig_7.nii'))
# A=np.expand_dims(,axis=0)
X=input[0:50,0:50,0:50]
Y=input[10:60,10:60,10:60]
input=np.stack((X,Y),axis=-1)
input=np.expand_dims(input,axis=0)
img=tf.placeholder(tf.float32,shape=[1,50,50,50,2])
# trainable=tf.placeholder(tf.bool)

# upsampling3d=upsampling()
# ups1 = upsampling3d.upsampling3d(img,
#                                  'unet_3dUS1')
sess=tf.Session()
# sess.run(tf.initialize_variables(tf.all_variables()))

# US=sess.run(ups1,feed_dict={img:input})


# input=np.expand_dims(input,axis=0)
img=tf.placeholder(tf.float32,shape=[1,50,50,50,2])
DS=ds.downsampler(img, down_scale=2, kernel_name='bspline', normalize_kernel=True, a=-.5, default_pixel_value=0)

result=sess.run(DS,feed_dict={img:input})
# plt.imshow(X[21,:,:],cmap='gray')
# plt.imshow(Y[21,:,:],cmap='gray')
# plt.imshow(result[0,10,:,:,0],cmap='gray')
# plt.imshow(result[0,10,:,:,1],cmap='gray')
# plt.show()
print('a')

