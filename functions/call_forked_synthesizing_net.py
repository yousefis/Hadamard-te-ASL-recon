import logging
import os
import time
import matplotlib.pyplot as plt
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
from functions.utils.gradients import gradients
import SimpleITK as sitk
# --------------------------------------------------------------------------------------------------------
class forked_synthesizing_net:
    def __init__(self,data,  sample_no,validation_samples,no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag,img_name,label_name,torso_tag,log_tag,min_range,max_range,
                 Logs,fold,newdataset=False):
        settings.init()
        # ==================================
        self.train_tag=train_tag
        self.validation_tag=validation_tag
        self.test_tag=test_tag
        self.img_name=img_name
        self.label_name=label_name
        self.torso_tag=torso_tag
        self.data=data
        self.gradients=gradients

        # ==================================
        settings.validation_totalimg_patch=validation_samples

        # ==================================
        self.learning_decay = .95
        self.learning_rate = 1E-5
        self.beta_rate = 0.05

        self.img_padded_size = 519
        self.seg_size = 505
        self.min_range = min_range
        self.max_range = max_range

        self.label_patchs_size =63
        self.patch_window = 77#89
        self.sample_no = sample_no
        self.batch_no =7
        self.batch_no_validation = 7
        self.validation_samples = validation_samples
        self.display_train_step = 25
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance=_loss_func()
        Server='shark'
        if Server == 'DL':
            self.parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
            if newdataset==True:
                self.data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation00_low/'
            else:
                self.data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/'
        else:
            self.parent_path = '/exports/lkeb-hpc/syousefi/Code/'
            if newdataset == True:
                self.data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation00_low/'
            else:
                self.data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'
        self.Logs=Logs

        self.no_sample_per_each_itr=no_sample_per_each_itr


        self.log_ext = log_tag
        self.LOGDIR = self.parent_path+self.Logs + self.log_ext + '/'
        self.chckpnt_dir = self.parent_path+self.Logs + self.log_ext + '/unet_checkpoints/'

        self.fold=fold


        logger.set_log_file(self.parent_path + self.Logs + self.log_ext + '/log_file' + str(fold))

    def save_file(self,file_name,txt):
        with open(file_name, 'a') as file:
            file.write(txt)





    def run_net(self):
        _rd = _read_data(data=self.data,
                         img_name=self.img_name, label_name=self.label_name,dataset_path=self.data_path)

        self.alpha_coeff=1
        '''read path of the images for train, test, and validation'''
        train_data, validation_data, test_data=_rd.read_data_path()

        # ======================================
        bunch_of_images_no=20
        sample_no=40
        _image_class_vl = image_class(validation_data,
                                      bunch_of_images_no=bunch_of_images_no,
                                      is_training=0,
                                      patch_window=self.patch_window,
                                      sample_no_per_bunch=sample_no,
                                      label_patch_size=self.label_patchs_size,
                                      validation_total_sample=self.validation_samples)


        _patch_extractor_thread_vl = _extractor_thread(_image_class=_image_class_vl,
                                                       patch_window=self.patch_window,
                                                       label_patchs_size=self.label_patchs_size,

                                                       mutex=settings.mutex,
                                                       is_training=0,
                                                       vl_sample_no=self.validation_samples
                                                       )
        _fill_thread_vl = fill_thread(data=validation_data,
                                      _image_class=_image_class_vl,
                                      sample_no=sample_no,
                                      total_sample_no=self.validation_samples,
                                      label_patchs_size=self.label_patchs_size,
                                      mutex=settings.mutex,
                                      is_training=0,
                                      patch_extractor=_patch_extractor_thread_vl,
                                      fold=self.fold)


        _fill_thread_vl.start()
        _patch_extractor_thread_vl.start()
        _read_thread_vl = read_thread(_fill_thread_vl,
                                      mutex=settings.mutex,
                                      validation_sample_no=self.validation_samples,
                                      is_training=0)
        _read_thread_vl.start()
        # ======================================
        bunch_of_images_no = 20
        sample_no=40
        _image_class = image_class(train_data
                                   , bunch_of_images_no=bunch_of_images_no,
                                   is_training=1,
                                   patch_window=self.patch_window,
                                   sample_no_per_bunch=sample_no,
                                   label_patch_size=self.label_patchs_size,
                                    validation_total_sample = 0)



        patch_extractor_thread = _extractor_thread(_image_class=_image_class,
                                                   patch_window=self.patch_window,
                                                   label_patchs_size=self.label_patchs_size,
                                                   mutex=settings.mutex, is_training=1)
        _fill_thread = fill_thread(train_data,
                                   _image_class,
                                   sample_no=sample_no, total_sample_no=self.sample_no,
                                   label_patchs_size=self.label_patchs_size,
                                   mutex=settings.mutex,
                                   is_training=1,
                                   patch_extractor=patch_extractor_thread,
                                   fold=self.fold)

        _fill_thread.start()
        patch_extractor_thread.start()

        _read_thread = read_thread(_fill_thread, mutex=settings.mutex, is_training=1)
        _read_thread.start()
        # ======================================
        # pre_bn=tf.placeholder(tf.float32,shape=[None,None,None,None,None])
        # image=tf.placeholder(tf.float32,shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window,1])
        # label=tf.placeholder(tf.float32,shape=[self.batch_no_validation,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size,2])
        # loss_coef=tf.placeholder(tf.float32,shape=[self.batch_no_validation,1,1,1])
        #
        img_row1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        img_row8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])

        label1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label9 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label10 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label11 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label12 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label13 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        label14 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])

        # img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        #
        # label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        # label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])


        is_training = tf.placeholder(tf.bool, name='is_training')
        input_dim = tf.placeholder(tf.int32, name='input_dim')

        forked_densenet=_forked_densenet()

        y,img_row1, img_row2, img_row3, img_row4,\
                 img_row5, img_row6, img_row7, img_row8=\
            forked_densenet.densenet( img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4, img_row5=img_row5,
                     img_row6=img_row6, img_row7=img_row7, img_row8=img_row8,input_dim=input_dim,is_training=is_training)
        #=================================================================================
        #=================================================================================
        # with tf.variable_scope('summary_debug'):
        # tf.summary.image('0_stackconcat0', level_ds1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis], 3)
        # tf.summary.image('0_stackconcat1', level_ds1[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis], 3)
        # tf.summary.image('0_stackconcat2', level_ds1[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis], 3)
        # tf.summary.image('0_stackconcat3', level_ds1[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis], 3)
        # tf.summary.image('0_stackconcat4', level_ds1[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis], 3)
        # tf.summary.image('0_stackconcat5', level_ds1[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis], 3)
        # tf.summary.image('0_stackconcat6', level_ds1[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis], 3)
        # tf.summary.image('0_stackconcat7', level_ds1[:, int(self.label_patchs_size / 2), :, :, 7, np.newaxis], 3)
        #
        # tf.summary.image('1_level_ds1_00', level_ds1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_01', level_ds1[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_02', level_ds1[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_03', level_ds1[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_04', level_ds1[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_05', level_ds1[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_06', level_ds1[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_07', level_ds1[:, int(self.label_patchs_size / 2), :, :, 7, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_08', level_ds1[:, int(self.label_patchs_size / 2), :, :, 8, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_09', level_ds1[:, int(self.label_patchs_size / 2), :, :, 9, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_10', level_ds1[:, int(self.label_patchs_size / 2), :, :, 10, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_11', level_ds1[:, int(self.label_patchs_size / 2), :, :, 11, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_12', level_ds1[:, int(self.label_patchs_size / 2), :, :, 12, np.newaxis], 3)
        # tf.summary.image('1_level_ds1_13', level_ds1[:, int(self.label_patchs_size / 2), :, :, 13, np.newaxis], 3)
        #
        # tf.summary.image('2_pool1_00', level_ds1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis], 3)
        # tf.summary.image('2_pool1_01', level_ds1[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis], 3)
        # tf.summary.image('2_pool1_02', level_ds1[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis], 3)
        # tf.summary.image('2_pool1_03', level_ds1[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis], 3)
        # tf.summary.image('2_pool1_04', level_ds1[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis], 3)
        # tf.summary.image('2_pool1_05', level_ds1[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis], 3)
        # tf.summary.image('2_pool1_06', level_ds1[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis], 3)
        # tf.summary.image('2_pool1_07', level_ds1[:, int(self.label_patchs_size / 2), :, :, 7, np.newaxis], 3)
        # tf.summary.image('2_pool1_08', level_ds1[:, int(self.label_patchs_size / 2), :, :, 8, np.newaxis], 3)
        # tf.summary.image('2_pool1_09', level_ds1[:, int(self.label_patchs_size / 2), :, :, 9, np.newaxis], 3)
        # tf.summary.image('2_pool1_10', level_ds1[:, int(self.label_patchs_size / 2), :, :, 10, np.newaxis], 3)
        # tf.summary.image('2_pool1_11', level_ds1[:, int(self.label_patchs_size / 2), :, :, 11, np.newaxis], 3)
        # tf.summary.image('2_pool1_12', level_ds1[:, int(self.label_patchs_size / 2), :, :, 12, np.newaxis], 3)
        # tf.summary.image('2_pool1_13', level_ds1[:, int(self.label_patchs_size / 2), :, :, 13, np.newaxis], 3)



        #=================================================================================
        #=================================================================================



        # with tf.variable_scope('summary_output_gt'):
        y_dirX = ((y[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]))
        y_dirX1 = ((y[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis]))
        y_dirX2 = ((y[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis]))
        y_dirX3 = ((y[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis]))
        y_dirX4 = ((y[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis]))
        y_dirX5 = ((y[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis]))
        y_dirX6 = ((y[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis]))
        y_dirX7 = ((y[:, int(self.label_patchs_size / 2), :, :, 7, np.newaxis]))
        y_dirX8 = ((y[:, int(self.label_patchs_size / 2), :, :, 8, np.newaxis]))
        y_dirX9 = ((y[:, int(self.label_patchs_size / 2), :, :, 9, np.newaxis]))
        y_dirX10 = ((y[:, int(self.label_patchs_size / 2), :, :, 10, np.newaxis]))
        y_dirX11 = ((y[:, int(self.label_patchs_size / 2), :, :, 11, np.newaxis]))
        y_dirX12 = ((y[:, int(self.label_patchs_size / 2), :, :, 12, np.newaxis]))
        y_dirX13 = ((y[:, int(self.label_patchs_size / 2), :, :, 13, np.newaxis]))

        label_dirX1 = (label1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX2 = (label2[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX3 = (label3[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX4 = (label4[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX5 = (label5[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX6 = (label6[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX7 = (label7[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX8 = (label8[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX9 = (label9[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX10 = (label10[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX11 = (label11[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX12 = (label12[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX13 = (label13[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        label_dirX14 = (label14[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])
        tf.summary.image('perfusion0',y_dirX ,3)
        tf.summary.image('perfusion1',y_dirX1 ,3)
        tf.summary.image('perfusion2',y_dirX2 ,3)
        tf.summary.image('perfusion3',y_dirX3 ,3)
        tf.summary.image('perfusion4',y_dirX4 ,3)
        tf.summary.image('perfusion5',y_dirX5 ,3)
        tf.summary.image('perfusion6',y_dirX6 ,3)
        tf.summary.image('angiography0',y_dirX7 ,3)
        tf.summary.image('angiography1',y_dirX8 ,3)
        tf.summary.image('angiography2',y_dirX9 ,3)
        tf.summary.image('angiography3',y_dirX10 ,3)
        tf.summary.image('angiography4',y_dirX11 ,3)
        tf.summary.image('angiography5',y_dirX12 ,3)
        tf.summary.image('angiography6',y_dirX13 ,3)

        tf.summary.image('GT_perfusion0', label_dirX1,3)
        tf.summary.image('GT_perfusion1', label_dirX2,3)
        tf.summary.image('GT_perfusion2', label_dirX3,3)
        tf.summary.image('GT_perfusion3', label_dirX4,3)
        tf.summary.image('GT_perfusion4', label_dirX5,3)
        tf.summary.image('GT_perfusion5', label_dirX6,3)
        tf.summary.image('GT_perfusion6', label_dirX7,3)
        tf.summary.image('GT_angiography0', label_dirX8,3)
        tf.summary.image('GT_angiography1', label_dirX9,3)
        tf.summary.image('GT_angiography2', label_dirX10,3)
        tf.summary.image('GT_angiography3', label_dirX11,3)
        tf.summary.image('GT_angiography4', label_dirX12,3)
        tf.summary.image('GT_angiography5', label_dirX13,3)
        tf.summary.image('GT_angiography6', label_dirX14,3)



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        log_extttt=''
        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' + log_extttt,graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' + log_extttt, graph=sess.graph)
        saver=tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        utils.backup_code(self.LOGDIR)

        '''AdamOptimizer:'''
        with tf.name_scope('averaged_mean_squared_error'):#
            [_loss, _ssim, _huber, _ssim_angio, _ssim_perf, _huber_angio, _huber_perf,perf_loss,angio_loss]= self.loss_instance.averaged_SSIM_huber(label1=label1,
                                                                 label2=label2,
                                                                 label3=label3,
                                                                 label4=label4,
                                                                 label5=label5,
                                                                 label6=label6,
                                                                 label7=label7,
                                                                 label8=label8,
                                                                 label9=label9,
                                                                 label10=label10,
                                                                 label11=label11,
                                                                 label12=label12,
                                                                 label13=label13,
                                                                 label14=label14,
                                                                 logit1=y[:, :, :, :, 0,np.newaxis],
                                                                 logit2=y[:, :, :, :, 1,np.newaxis],
                                                                 logit3=y[:, :, :, :, 2,np.newaxis],
                                                                 logit4=y[:, :, :, :, 3,np.newaxis],
                                                                 logit5=y[:, :, :, :, 4,np.newaxis],
                                                                 logit6=y[:, :, :, :, 5,np.newaxis],
                                                                 logit7=y[:, :, :, :, 6,np.newaxis],
                                                                 logit8=y[:, :, :, :, 7,np.newaxis],
                                                                 logit9=y[:, :, :, :, 8,np.newaxis],
                                                                 logit10=y[:, :, :, :, 9,np.newaxis],
                                                                 logit11=y[:, :, :, :, 10,np.newaxis],
                                                                 logit12=y[:, :, :, :, 11,np.newaxis],
                                                                 logit13=y[:, :, :, :, 12,np.newaxis],
                                                                 logit14=y[:, :, :, :, 13,np.newaxis]
                                                                 )
            cost = tf.reduce_mean(_loss, name="cost")
            ssim_cost = tf.reduce_mean(_ssim, name="ssim_cost")
            huber_cost = tf.reduce_mean(_huber, name="huber_cost")

            ssim_angio = tf.reduce_mean(_ssim_angio, name="ssim_angio")
            ssim_perf = tf.reduce_mean(_ssim_perf, name="ssim_perf")
            huber_angio = tf.reduce_mean(_huber_angio, name="huber_angio")
            huber_perf = tf.reduce_mean(_huber_perf, name="huber_perf")


        # ========================================================================
        ave_loss = tf.placeholder(tf.float32, name='loss')
        ave_loss_perf = tf.placeholder(tf.float32, name='loss_perf')
        ave_loss_angio = tf.placeholder(tf.float32, name='loss_angio')

        average_gradient_perf = tf.placeholder(tf.float32, name='grad_ave_perf')
        average_gradient_angio = tf.placeholder(tf.float32, name='grad_ave_angio')




        tf.summary.scalar("Loss/ave_loss", ave_loss)
        tf.summary.scalar("Loss/ave_loss_perf", ave_loss_perf)
        tf.summary.scalar("Loss/ave_loss_angio", ave_loss_angio)

        tf.summary.scalar("huber/huber_angio", huber_angio)
        tf.summary.scalar("ssim/ssim_angio", ssim_angio)
        tf.summary.scalar("ssim/ssim_perf", ssim_perf)
        tf.summary.scalar("huber/huber_perf", huber_perf)



        tf.summary.scalar("ssim/ssim", ssim_cost)
        tf.summary.scalar("huber/huber", huber_cost)

        tf.summary.scalar('gradients/avegare_perfusion', average_gradient_perf)
        tf.summary.scalar('gradients/avegare_angiography', average_gradient_angio)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)



        sess.run(tf.global_variables_initializer())
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        grad_p,grad_a= self.gradients.comput_gradients(_loss, perf_loss, angio_loss, var_list)


        logging.debug('total number of variables %s' % (
        np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        summ=tf.summary.merge_all()
        loadModel = 0
        point = 0

        itr1 = 0
        if loadModel:
            chckpnt_dir=''
            ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            point=np.int16(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            itr1=point

        '''loop for epochs'''
        for epoch in range(self.total_epochs):
            while self.no_sample_per_each_itr*int(point/self.no_sample_per_each_itr)<self.sample_no:
                print('0')
                print("epoch #: %d" %(epoch))
                startTime = time.time()

                step = 0
                # =============validation================
                if itr1 % self.display_validation_step ==0:
                    '''Validation: '''
                    loss_validation = 0
                    loss_validation_perf = 0
                    loss_validation_angio = 0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation=0

                while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):
                        [crush,noncrush,perf,angio]=_image_class_vl.return_patches_vl( validation_step * self.batch_no_validation,
                                                                                     ( validation_step + 1) *self.batch_no_validation,is_tr=False)
                        if (len(angio)<self.batch_no_validation ) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue
                        plt.imshow(perf[0,0,30,:,:,0])
                        start=time.time()
                        [loss_vali,perf_loss_vali, angio_loss_vali,] = sess.run([cost,perf_loss, angio_loss],
                                                         feed_dict={img_row1:crush[:,0,:,:,:,:],
                                                                    img_row2:noncrush[:,1,:,:,:,:],
                                                                    img_row3:crush[:,2,:,:,:,:],
                                                                    img_row4:noncrush[:,3,:,:,:,:],
                                                                    img_row5 : crush[:,4,:,:,:,:],
                                                                    img_row6: noncrush[:,5,:,:,:,:],
                                                                    img_row7 : crush[:,6,:,:,:,:],
                                                                    img_row8 : noncrush[:,7,:,:,:,:],
                                                                    label1 : perf[:,0,:,:,:,:],
                                                                    label2 : perf[:,1,:,:,:,:],
                                                                    label3 : perf[:,2,:,:,:,:],
                                                                    label4 : perf[:,3,:,:,:,:],
                                                                    label5 : perf[:,4,:,:,:,:],
                                                                    label6 : perf[:,5,:,:,:,:],
                                                                    label7 : perf[:,6,:,:,:,:],
                                                                    label8 : angio[:,0,:,:,:,:],
                                                                    label9 : angio[:,1,:,:,:,:],
                                                                    label10 : angio[:,2,:,:,:,:],
                                                                    label11 : angio[:,3,:,:,:,:],
                                                                    label12 : angio[:,4,:,:,:,:],
                                                                    label13 : angio[:,5,:,:,:,:],
                                                                    label14 : angio[:,6,:,:,:,:],
                                                                    is_training:False,
                                                                    input_dim:self.patch_window,
                                                                    ave_loss: -1,
                                                                    ave_loss_perf:-1,
                                                                    ave_loss_angio:-1,
                                                                    average_gradient_perf:-1,
                                                                    average_gradient_angio:-1
                                                                    })
                        end=time.time()

                        loss_validation += loss_vali
                        loss_validation_perf+=perf_loss_vali
                        loss_validation_angio+=angio_loss_vali
                        # if totat_grad_p_vali==0:
                        #     totat_grad_p_vali=grad_p_vali
                        #     totat_grad_a_vali=grad_a_vali
                        # else:
                        #     totat_grad_p_vali+=grad_p_vali
                        #     totat_grad_a_vali+=grad_a_vali
                        #
                        # if validation_step%20==0:
                        #     sum_totat_grad_p_vali = self.sum_gradients(totat_grad_p_vali)
                        #     sum_totat_grad_a_vali = self.sum_gradients(totat_grad_a_vali)
                        #     self.average_gradients()



                        validation_step += 1
                        if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                            print('nan problem')
                        process = psutil.Process(os.getpid())
                        print(
                            '%d - > %d:   loss_validation: %f, gpu_elapse_time: %4s, memory_percent: %4s' % (
                                validation_step, validation_step * self.batch_no_validation
                                , loss_vali,str(end-start), str(process.memory_percent()),
                            ))

                settings.queue_isready_vl = False
                acc_validation = acc_validation / (validation_step)
                loss_validation = loss_validation / (validation_step)
                loss_validation_perf = loss_validation_perf / (validation_step)
                loss_validation_angio = loss_validation_angio / (validation_step)
                dsc_validation = dsc_validation / (validation_step)





                if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                    print('nan problem')
                _fill_thread_vl.kill_thread()
                print('******Validation, step: %d , accuracy: %.4f, loss: %f*******' % (
                itr1, acc_validation, loss_validation))
                [sum_validation] = sess.run([summ],
                                            feed_dict={img_row1:crush[:,0,:,:,:,:],
                                                                    img_row2:noncrush[:,1,:,:,:,:],
                                                                    img_row3:crush[:,2,:,:,:,:],
                                                                    img_row4:noncrush[:,3,:,:,:,:],
                                                                    img_row5 : crush[:,4,:,:,:,:],
                                                                    img_row6: noncrush[:,5,:,:,:,:],
                                                                    img_row7 : crush[:,6,:,:,:,:],
                                                                    img_row8 : noncrush[:,7,:,:,:,:],
                                                                    label1 : perf[:,0,:,:,:,:],
                                                                    label2 : perf[:,1,:,:,:,:],
                                                                    label3 : perf[:,2,:,:,:,:],
                                                                    label4 : perf[:,3,:,:,:,:],
                                                                    label5 : perf[:,4,:,:,:,:],
                                                                    label6 : perf[:,5,:,:,:,:],
                                                                    label7 : perf[:,6,:,:,:,:],
                                                                    label8 : angio[:,0,:,:,:,:],
                                                                    label9 : angio[:,1,:,:,:,:],
                                                                    label10 : angio[:,2,:,:,:,:],
                                                                    label11 : angio[:,3,:,:,:,:],
                                                                    label12 : angio[:,4,:,:,:,:],
                                                                    label13 : angio[:,5,:,:,:,:],
                                                                    label14 : angio[:,6,:,:,:,:],
                                                                    is_training: False,
                                                                    input_dim: self.patch_window,
                                                                    ave_loss: loss_validation,
                                                                    ave_loss_perf: loss_validation_perf,
                                                                    ave_loss_angio: loss_validation_angio,
                                                                    average_gradient_perf: -1,
                                                                    average_gradient_angio:-1
                                                                       })
                validation_writer.add_summary(sum_validation, point)
                print('end of validation---------%d' % (point))


                '''-----------------loop for training batches---------------------------'''
                while(step*self.batch_no<self.no_sample_per_each_itr):

                    # [train_CT_image_patchs, train_GTV_label, train_Penalize_patch,loss_coef_weights] = _image_class.return_patches( self.batch_no)

                    [crush, noncrush, perf, angio] = _image_class.return_patches_tr(self.batch_no)

                    if (len(angio)<self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue


                    if itr1 % self.display_train_step == 0:
                        '''train: '''
                        train_step=0
                        average_loss_tr=0
                        average_loss_tr_perf=0
                        average_loss_tr_angio=0

                    while (train_step <self.display_train_step):
                        if train_step==0:
                            totat_grad_p_tr = 0
                            totat_grad_a_tr = 0
                        # grad_p_tr=0
                        # grad_a_tr=0
                        start1 = time.time()
                        [grad_p_tr, grad_a_tr, ] = sess.run([grad_p, grad_a,],
                                                       feed_dict={img_row1: crush[:, 0, :, :, :, :],
                                                                  img_row2: noncrush[:, 1, :, :, :, :],
                                                                  img_row3: crush[:, 2, :, :, :, :],
                                                                  img_row4: noncrush[:, 3, :, :, :, :],
                                                                  img_row5: crush[:, 4, :, :, :, :],
                                                                  img_row6: noncrush[:, 5, :, :, :, :],
                                                                  img_row7: crush[:, 6, :, :, :, :],
                                                                  img_row8: noncrush[:, 7, :, :, :, :],
                                                                  label1: perf[:, 0, :, :, :, :],
                                                                  label2: perf[:, 1, :, :, :, :],
                                                                  label3: perf[:, 2, :, :, :, :],
                                                                  label4: perf[:, 3, :, :, :, :],
                                                                  label5: perf[:, 4, :, :, :, :],
                                                                  label6: perf[:, 5, :, :, :, :],
                                                                  label7: perf[:, 6, :, :, :, :],
                                                                  label8: angio[:, 0, :, :, :, :],
                                                                  label9: angio[:, 1, :, :, :, :],
                                                                  label10: angio[:, 2, :, :, :, :],
                                                                  label11: angio[:, 3, :, :, :, :],
                                                                  label12: angio[:, 4, :, :, :, :],
                                                                  label13: angio[:, 5, :, :, :, :],
                                                                  label14: angio[:, 6, :, :, :, :],
                                                                  is_training: False,
                                                                  input_dim: self.patch_window,
                                                                  ave_loss: -1,
                                                                  ave_loss_perf: -1,
                                                                  ave_loss_angio: -1,
                                                                  average_gradient_perf: -1,
                                                                  average_gradient_angio: -1
                                                                  })
                        end1 = time.time()
                        start = time.time()
                        [ loss_train1,perf_loss_tr, angio_loss_tr,
                          optimizing,out,] = sess.run([ cost,perf_loss, angio_loss,
                                                        optimizer,y,],
                                                                  feed_dict={img_row1:crush[:,0,:,:,:,:],
                                                                        img_row2:noncrush[:,1,:,:,:,:],
                                                                        img_row3:crush[:,2,:,:,:,:],
                                                                        img_row4:noncrush[:,3,:,:,:,:],
                                                                        img_row5 : crush[:,4,:,:,:,:],
                                                                        img_row6: noncrush[:,5,:,:,:,:],
                                                                        img_row7 : crush[:,6,:,:,:,:],
                                                                        img_row8 : noncrush[:,7,:,:,:,:],
                                                                        label1 : perf[:,0,:,:,:,:],
                                                                        label2 : perf[:,1,:,:,:,:],
                                                                        label3 : perf[:,2,:,:,:,:],
                                                                        label4 : perf[:,3,:,:,:,:],
                                                                        label5 : perf[:,4,:,:,:,:],
                                                                        label6 : perf[:,5,:,:,:,:],
                                                                        label7 : perf[:,6,:,:,:,:],
                                                                        label8 : angio[:,0,:,:,:,:],
                                                                        label9 : angio[:,1,:,:,:,:],
                                                                        label10 : angio[:,2,:,:,:,:],
                                                                        label11 : angio[:,3,:,:,:,:],
                                                                        label12 : angio[:,4,:,:,:,:],
                                                                        label13 : angio[:,5,:,:,:,:],
                                                                        label14 : angio[:,6,:,:,:,:],
                                                                        is_training: False,
                                                                        input_dim: self.patch_window,
                                                                         ave_loss: -1,
                                                                         ave_loss_perf: -1,
                                                                         ave_loss_angio: -1,
                                                                         average_gradient_perf: -1,
                                                                         average_gradient_angio: -1
                                                                        })
                        end = time.time()
                        if totat_grad_p_tr == 0:
                            totat_grad_p_tr = grad_p_tr
                            totat_grad_a_tr = grad_a_tr
                        else:
                            totat_grad_p_tr += grad_p_tr
                            totat_grad_a_tr += grad_a_tr
                        average_loss_tr += loss_train1
                        average_loss_tr_perf += perf_loss_tr
                        average_loss_tr_angio += angio_loss_tr
                        point = point +1
                        print(
                            'point: %d, step*self.batch_no:%f , training_elapsed_time: %4s,gradient_elapsed_time: %4s,'
                            ' loss_train_point:%f,memory_percent: %4s' % (
                                int((point)),
                                step * self.batch_no,str(end-start) ,str(end1-start1),loss_train1,
                                str(process.memory_percent())))

                        train_step+=1
                    average_loss_tr /= self.display_train_step
                    average_loss_tr_perf /=  self.display_train_step
                    average_loss_tr_angio /=  self.display_train_step
                    sum_totat_grad_p_tr=self.gradients.sum_gradients(totat_grad_p_tr)
                    sum_totat_grad_a_tr=self.gradients.sum_gradients(totat_grad_a_tr)

                    [sum_train] = sess.run([summ],
                                           feed_dict={img_row1:crush[:,0,:,:,:,:],
                                                        img_row2:noncrush[:,1,:,:,:,:],
                                                        img_row3:crush[:,2,:,:,:,:],
                                                        img_row4:noncrush[:,3,:,:,:,:],
                                                        img_row5 : crush[:,4,:,:,:,:],
                                                        img_row6: noncrush[:,5,:,:,:,:],
                                                        img_row7 : crush[:,6,:,:,:,:],
                                                        img_row8 : noncrush[:,7,:,:,:,:],
                                                        label1 : perf[:,0,:,:,:,:],
                                                        label2 : perf[:,1,:,:,:,:],
                                                        label3 : perf[:,2,:,:,:,:],
                                                        label4 : perf[:,3,:,:,:,:],
                                                        label5 : perf[:,4,:,:,:,:],
                                                        label6 : perf[:,5,:,:,:,:],
                                                        label7 : perf[:,6,:,:,:,:],
                                                        label8 : angio[:,0,:,:,:,:],
                                                        label9 : angio[:,1,:,:,:,:],
                                                        label10 : angio[:,2,:,:,:,:],
                                                        label11 : angio[:,3,:,:,:,:],
                                                        label12 : angio[:,4,:,:,:,:],
                                                        label13 : angio[:,5,:,:,:,:],
                                                        label14 : angio[:,6,:,:,:,:],
                                                        is_training: False,
                                                        input_dim: self.patch_window,
                                                        ave_loss: average_loss_tr,
                                                        ave_loss_perf: average_loss_tr_perf,
                                                        ave_loss_angio: average_loss_tr_angio,
                                                        average_gradient_perf:sum_totat_grad_p_tr,
                                                        average_gradient_angio:sum_totat_grad_a_tr
                                                        })

                    train_writer.add_summary(sum_train,point)
                    step = step + self.display_train_step



                    process = psutil.Process(os.getpid())
                    print(
                        'point: %d, step*self.batch_no:%f , LR: %.15f, loss_train1:%f,memory_percent: %4s' % (
                        int((point)),
                        step * self.batch_no, self.learning_rate,  loss_train1,
                        str(process.memory_percent())))
                    point=int((point))#(self.no_sample_per_each_itr/self.batch_no)*itr1+step

                    if point%100==0:
                        '''saveing model inter epoch'''
                        chckpnt_path = os.path.join(self.chckpnt_dir,
                                                    ('unet_inter_epoch%d_point%d.ckpt' % (epoch, point)))
                        saver.save(sess, chckpnt_path, global_step=point)



                    itr1 = itr1 + self.display_train_step
                    # point=point+self.display_train_step

            endTime = time.time()
            #==============end of epoch:
            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'unet.ckpt')
            saver.save(sess, chckpnt_path, global_step=epoch)


            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))


