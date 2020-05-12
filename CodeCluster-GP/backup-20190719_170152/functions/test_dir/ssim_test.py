from functions.loss.ssim_loss import *
import SimpleITK as sitk
import numpy as np
path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/04_PP2_04_cinema2/decoded_crush/'
I1=np.expand_dims(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_0.nii')),0),-1)
I2=np.expand_dims(np.expand_dims(sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_0.nii')),0),-1)
in1=tf.placeholder(tf.float32,[1,None,None,None,1])
in2=tf.placeholder(tf.float32,[1,None,None,None,1])
[loss,ssim_map] = SSIM(in1,in2)
sess=tf.Session()
[sim,ssim_map] = sess.run([loss,ssim_map],feed_dict={in1: I1, in2:I2})
print(sim)