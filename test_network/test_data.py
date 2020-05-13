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
data_path=''
log_tag='synth-'+str(fold)
min_range=-1000
max_range=3000
Logs='ASL_LOG/debug_Log/',
fold=fold
bunch_of_images_no=20
patch_window=77
label_patchs_size=77
validation_samples=100
_rd = _read_data(data=data,
                         img_name=img_name, label_name=label_name,dataset_path=data_path)

alpha_coeff=1
'''read path of the images for train, test, and validation'''
train_data, validation_data, test_data=_rd.read_data_path()
_image_class_vl = image_class(validation_data,
                              bunch_of_images_no=bunch_of_images_no,
                              is_training=0,
                              patch_window=patch_window,
                              sample_no_per_bunch=sample_no,
                              label_patch_size=label_patchs_size,
                              validation_total_sample=validation_samples)
sitk.ReadImage(train_data)
plt.imshow()
