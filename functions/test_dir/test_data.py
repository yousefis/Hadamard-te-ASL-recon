import logging
import os
import time

import numpy as np
import psutil
import tensorflow as tf
from functions.image_reader.read_data import _read_data

import functions.utils.utils as utils
import functions.utils.logger as logger
import functions.utils.settings as settings
from functions.image_reader.image_class import image_class
from functions.loss.loss_fun import _loss_func
from functions.network.forked_densenet import _forked_densenet

# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread
import  SimpleITK as sitk
import matplotlib.pyplot as plt
data=2
fold=-1
sample_no=2000000
validation_samples=396
no_sample_per_each_itr=1000
train_tag=''
validation_tag=''
test_tag=''
img_name=''
label_name=''
torso_tag=''
data_path='/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'
log_tag='synth-'+str(fold)
min_range=-1000
max_range=3000
Logs='ASL_LOG/debug_Log/',
fold=fold
bunch_of_images_no=20
patch_window=77
label_patchs_size=77
_rd = _read_data(data=data,
                         img_name=img_name, label_name=label_name,dataset_path=data_path)

alpha_coeff=1
'''read path of the images for train, test, and validation'''
train_data, validation_data, test_data=_rd.read_data_path()

sitk_I=sitk.ReadImage(train_data[100][0][0])

I=sitk.GetArrayFromImage(sitk_I)
# noisyI=I + np.reshape(np.random.normal(.2,0, np.size(I)),[60,85,85])
# sitknoisyI=sitk.GetImageFromArray(noisyI)
# sitknoisyI.SetSpacing(sitk_I.GetSpacing())
# sitknoisyI.SetOrigin(sitk_I.GetOrigin())
# sitknoisyI.SetDirection(sitk_I.GetDirection())
seed=200
s=tf.Session()
shape=(50,50,50)
num=10
II=tf.placeholder(tf.float32, shape=[None,None,None])

salty_img=(tf.round(tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32, seed=seed)))
rnd_stdev = s.run(tf.random_uniform(shape=[1], minval=0, maxval=5, dtype=tf.int32,
                                                seed=seed)) / 200  # , seed=int(time.time())))
rnd_mean = s.run(tf.random_uniform(shape=[1], minval=0, maxval=5, dtype=tf.int32,
                                   seed=seed)) / 100  # , seed=int(time.time())))
# [,y[i],x[i]]

tfnoisyI=II + (tf.random_normal(tf.shape(II),
                                      mean=rnd_mean,
                                      stddev=rnd_stdev,
                                      seed=seed + 2,
                                      dtype=tf.float32))
noisyI= s.run(tfnoisyI,feed_dict={II:I})

sitknoisyI=sitk.GetImageFromArray(noisyI)
sitknoisyI.SetSpacing(sitk_I.GetSpacing())
sitknoisyI.SetOrigin(sitk_I.GetOrigin())
sitknoisyI.SetDirection(sitk_I.GetDirection())

sitk.WriteImage(sitknoisyI,'/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/test_data/noisy.mha')
sitk.WriteImage(sitk_I,'/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/test_data/image.mha')
plt.imshow(I[20,:,:])
print('xx')
