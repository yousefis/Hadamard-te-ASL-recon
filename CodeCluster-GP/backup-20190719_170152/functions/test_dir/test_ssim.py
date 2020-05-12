from functions.loss.ssim_loss import *
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
if __name__=='__main__':
    path = '/exports/lkeb-hpc/syousefi/Data/Simulated_data/BrainWeb_permutation7/04_PP4_04_cinema1/decoded_crush/'
    # I1 = sitk.GetArrayFromImage(sitk.ReadImage(path + 'crush_0.nii'))
    # I2 = sitk.GetArrayFromImage(sitk.ReadImage(path + 'crush_1.nii'))

    I1=np.ones((181,256,256))
    I2=np.zeros((181,256,256))

    X=tf.placeholder(tf.float32,(181,256,256,2))
    Y=tf.placeholder(tf.float32,(181,256,256,2))
    mssim, ssim_map=SSIM(X, Y, max_val=1.0)

    MSE = tf.losses.mean_squared_error(
        labels=X,
        predictions=Y)

    sess=tf.Session()
    mssim_res, ssim_map_res,mse= sess.run([mssim, ssim_map,MSE],
             feed_dict={X:np.stack((I1,I1),-1),Y:np.stack((I2,I2),-1)})


    print(np.sum(ssim_map_res),mse)



    z=40
    plt.imshow(I1[z, :, :])
    plt.figure()
    plt.imshow(I2[z, :, :])
    plt.figure()
    plt.imshow(1-ssim_map_res[z, :, :,0])
    plt.show()
    print('c')