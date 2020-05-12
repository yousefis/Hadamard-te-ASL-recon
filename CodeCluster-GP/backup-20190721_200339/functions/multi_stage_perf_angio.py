from functions.quantification_measure.quantifications import quantifications
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
from functions.loss.perceptual_loss.vgg.vgg_feature_maker import vgg_feature_maker
from functions.network.multi_stage_densenet import _multi_stage_densenet
# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread

from functions.utils.gradients import gradients
# --------------------------------------------------------------------------------------------------------
class multi_stage_net_perf_mri:
    def __init__(self, data, sample_no, validation_samples, no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag, img_name, label_name, torso_tag, log_tag, min_range, max_range,
                 Logs, fold, Server,newdataset=False):
        settings.init()
        self.loss_instance=_loss_func()
        # ==================================
        self.train_tag = train_tag
        self.validation_tag = validation_tag
        self.test_tag = test_tag
        self.img_name = img_name
        self.label_name = label_name
        self.torso_tag = torso_tag
        self.data = data
        self.display_train_step = 25
        # ==================================
        settings.validation_totalimg_patch = validation_samples
        self.gradients=gradients
        self.quantifications = quantifications()
        # ==================================
        self.learning_decay = .95
        self.learning_rate = 1E-5
        self.beta_rate = 0.05
        self.newdataset=newdataset

        self.img_padded_size = 519
        self.seg_size = 505
        self.min_range = min_range
        self.max_range = max_range

        self.label_patchs_size = 39  # 63
        self.patch_window = 53  # 77#89
        self.sample_no = sample_no
        self.batch_no = 6
        self.batch_no_validation = self.batch_no
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.fold = fold

        if Server == 'DL':
            self.parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
            self.data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation00_low/'

        else:
            self.parent_path = '/exports/lkeb-hpc/syousefi/Code/'

            self.data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation00_low/'
        self.Logs = Logs

        self.no_sample_per_each_itr = no_sample_per_each_itr

        self.log_ext = log_tag
        self.LOGDIR = self.parent_path + self.Logs + self.log_ext + '/'
        self.chckpnt_dir = self.parent_path + self.Logs + self.log_ext + '/unet_checkpoints/'

        logger.set_log_file(self.parent_path + self.Logs + self.log_ext + '/log_file' + str(fold))

    # def save_file(self, file_name, txt):
    #     with open(file_name, 'a') as file:
    #         file.write(txt)



    def run_net(self,loadModel = 0):

        # pre_bn=tf.placeholder(tf.float32,shape=[None,None,None,None,None])
        # image=tf.placeholder(tf.float32,shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window,1])
        # label=tf.placeholder(tf.float32,shape=[self.batch_no_validation,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size,2])
        # loss_coef=tf.placeholder(tf.float32,shape=[self.batch_no_validation,1,1,1])
        # ===================================================================================
        _rd = _read_data(data=self.data, train_tag=self.train_tag, validation_tag=self.validation_tag,
                         test_tag=self.test_tag,
                         img_name=self.img_name, label_name=self.label_name, torso_tag=self.torso_tag,
                         dataset_path=self.data_path,reverse=self.newdataset)

        self.alpha_coeff = 1
        '''read path of the images for train, test, and validation'''
        train_data, validation_data, test_data = _rd.read_data_path()

        # ======================================
        bunch_of_images_no = 20
        sample_no = 60
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
        sample_no = 60
        _image_class = image_class(train_data
                                   , bunch_of_images_no=bunch_of_images_no,
                                   is_training=1,
                                   patch_window=self.patch_window,
                                   sample_no_per_bunch=sample_no,
                                   label_patch_size=self.label_patchs_size,
                                   validation_total_sample=0)

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
        #
        _fill_thread.start()
        patch_extractor_thread.start()

        _read_thread = read_thread(_fill_thread, mutex=settings.mutex, is_training=1)
        _read_thread.start()
        # ===================================================================================
        img_row1 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row1')
        img_row2 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row2')
        img_row3 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row3')
        img_row4 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row4')
        img_row5 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row5')
        img_row6 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row6')
        img_row7 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row7')
        img_row8 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='img_row8')

        mri_ph = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],name='mri')
        segmentation = tf.placeholder(tf.float32,
                                      shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                             self.label_patchs_size, 1], name='segments')
        label1 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label1')
        label2 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label2')
        label3 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label3')
        label4 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label4')
        label5 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label5')
        label6 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label6')
        label7 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label7')
        label8 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label8')
        label9 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                   self.label_patchs_size, 1], name='label9')
        label10 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                    self.label_patchs_size, 1], name='label10')
        label11 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                    self.label_patchs_size, 1], name='label11')
        label12 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                    self.label_patchs_size, 1], name='label12')
        label13 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                    self.label_patchs_size, 1], name='label13')
        label14 = tf.placeholder(tf.float32, shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
                                                    self.label_patchs_size, 1], name='label14')

        # loss_placeholder = tf.placeholder(tf.float32,
        #                                   shape=[self.batch_no, self.label_patchs_size, self.label_patchs_size,
        #                                          self.label_patchs_size, 1])

        # img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row1')
        # img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row2')
        # img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row3')
        # img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row4')
        # img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row5')
        # img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row6')
        # img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row7')
        # img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row8')
        #
        # label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label1')
        # label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label2')
        # label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label3')
        # label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label4')
        # label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label5')
        # label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label6')
        # label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label7')
        # label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label8')
        # label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label9')
        # label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label10')
        # label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label11')
        # label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label12')
        # label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label13')
        # label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label14')

        is_training = tf.placeholder(tf.bool, name='is_training')
        input_dim = tf.placeholder(tf.int32, name='input_dim')

        # unet=_unet()
        multi_stage_densenet = _multi_stage_densenet()

        y,loss_upsampling11,loss_upsampling22 = multi_stage_densenet.multi_stage_densenet(img_row1=img_row1,
                              img_row2=img_row2,
                              img_row3=img_row3,
                              img_row4=img_row4,
                              img_row5=img_row5,
                              img_row6=img_row6,
                              img_row7=img_row7,
                              img_row8=img_row8,
                              input_dim=input_dim,
                              mri = mri_ph,
                              is_training=is_training)
        # self.vgg = vgg_feature_maker()
        # self.vgg_y0 = self.vgg.feed_img(loss_placeholder)
        # self.vgg_y0 = self.vgg.feed_img(y[:, :, :, :, 0]).copy()
        # self.vgg_y1 = self.vgg.feed_img(y[:, :, :, :, 1]).copy()
        # self.vgg_y2 = self.vgg.feed_img(y[:, :, :, :, 2]).copy()
        # self.vgg_y3 = self.vgg.feed_img(y[:, :, :, :, 3]).copy()
        # self.vgg_y4 = self.vgg.feed_img(y[:, :, :, :, 4]).copy()
        # self.vgg_y5 = self.vgg.feed_img(y[:, :, :, :, 5]).copy()
        # self.vgg_y6 = self.vgg.feed_img(y[:, :, :, :, 6]).copy()
        # self.vgg_y7 = self.vgg.feed_img(y[:, :, :, :, 7])
        # self.vgg_y8 = self.vgg.feed_img(y[:, :, :, :, 8])
        # self.vgg_y9 = self.vgg.feed_img(y[:, :, :, :, 9])
        # self.vgg_y10 = self.vgg.feed_img(y[:, :, :, :, 10])
        # self.vgg_y11 = self.vgg.feed_img(y[:, :, :, :, 11])
        # self.vgg_y12 = self.vgg.feed_img(y[:, :, :, :, 12])
        # self.vgg_y13 = self.vgg.feed_img(y[:, :, :, :, 13])

        # self.vgg_label0 = self.vgg.feed_img(label1[:,:,:,:,0]).copy()
        # self.vgg_label1 = self.vgg.feed_img(label2[:,:,:,:,0]).copy()
        # self.vgg_label2 = self.vgg.feed_img(label3[:,:,:,:,0]).copy()
        # self.vgg_label3 = self.vgg.feed_img(label4[:,:,:,:,0]).copy()
        # self.vgg_label4 = self.vgg.feed_img(label5[:,:,:,:,0]).copy()
        # self.vgg_label5 = self.vgg.feed_img(label6[:,:,:,:,0]).copy()
        # self.vgg_label6 = self.vgg.feed_img(label7[:,:,:,:,0]).copy()
        # self.vgg_label7 = self.vgg.feed_img(label8[:,:,:,:,0])
        # self.vgg_label8 = self.vgg.feed_img(label9[:,:,:,:,0])
        # self.vgg_label9 = self.vgg.feed_img(label10[:,:,:,:,0])
        # self.vgg_label10 = self.vgg.feed_img(label11[:,:,:,:,0])
        # self.vgg_label11 = self.vgg.feed_img(label12[:,:,:,:,0])
        # self.vgg_label12 = self.vgg.feed_img(label13[:,:,:,:,0])
        # self.vgg_label13 = self.vgg.feed_img(label14[:,:,:,:,0])

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

        tf.summary.image('out0', y_dirX, 3)
        tf.summary.image('out1', y_dirX1, 3)
        tf.summary.image('out2', y_dirX2, 3)
        tf.summary.image('out3', y_dirX3, 3)
        tf.summary.image('out4', y_dirX4, 3)
        tf.summary.image('out5', y_dirX5, 3)
        tf.summary.image('out6', y_dirX6, 3)
        tf.summary.image('out7', y_dirX7, 3)
        tf.summary.image('out8', y_dirX8, 3)
        tf.summary.image('out9', y_dirX9, 3)
        tf.summary.image('out10', y_dirX10, 3)
        tf.summary.image('out11', y_dirX11, 3)
        tf.summary.image('out12', y_dirX12, 3)
        tf.summary.image('out13', y_dirX13, 3)

        tf.summary.image('groundtruth1', label_dirX1, 3)
        tf.summary.image('groundtruth2', label_dirX2, 3)
        tf.summary.image('groundtruth3', label_dirX3, 3)
        tf.summary.image('groundtruth4', label_dirX4, 3)
        tf.summary.image('groundtruth5', label_dirX5, 3)
        tf.summary.image('groundtruth6', label_dirX6, 3)
        tf.summary.image('groundtruth7', label_dirX7, 3)
        tf.summary.image('groundtruth8', label_dirX8, 3)
        tf.summary.image('groundtruth9', label_dirX9, 3)
        tf.summary.image('groundtruth10', label_dirX10, 3)
        tf.summary.image('groundtruth11', label_dirX11, 3)
        tf.summary.image('groundtruth12', label_dirX12, 3)
        tf.summary.image('groundtruth13', label_dirX13, 3)
        tf.summary.image('groundtruth14', label_dirX14, 3)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' , graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' , graph=sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
        if loadModel==0:
            utils.backup_code(self.LOGDIR)
        labels = []
        labels.append(label1)
        labels.append(label2)
        labels.append(label3)
        labels.append(label4)
        labels.append(label5)
        labels.append(label6)
        labels.append(label7)
        labels.append(label8)
        labels.append(label9)
        labels.append(label10)
        labels.append(label11)
        labels.append(label12)
        labels.append(label13)
        labels.append(label14)

        logits = []
        logits.append(y[:, :, :, :, 0, np.newaxis])
        logits.append(y[:, :, :, :, 1, np.newaxis])
        logits.append(y[:, :, :, :, 2, np.newaxis])
        logits.append(y[:, :, :, :, 3, np.newaxis])
        logits.append(y[:, :, :, :, 4, np.newaxis])
        logits.append(y[:, :, :, :, 5, np.newaxis])
        logits.append(y[:, :, :, :, 6, np.newaxis])

        logits.append(y[:, :, :, :, 7, np.newaxis])
        logits.append(y[:, :, :, :, 8, np.newaxis])
        logits.append(y[:, :, :, :, 9, np.newaxis])
        logits.append(y[:, :, :, :, 10, np.newaxis])
        logits.append(y[:, :, :, :, 11, np.newaxis])
        logits.append(y[:, :, :, :, 12, np.newaxis])
        logits.append(y[:, :, :, :, 13, np.newaxis])

        stage1 = []
        stage1.append(loss_upsampling11[:, :, :, :, 0, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 1, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 2, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 3, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 4, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 5, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 6, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 7, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 8, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 9, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 10, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 11, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 12, np.newaxis])
        stage1.append(loss_upsampling11[:, :, :, :, 13, np.newaxis])

        stage2 = []
        stage2.append(loss_upsampling22[:, :, :, :, 0, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 1, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 2, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 3, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 4, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 5, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 6, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 7, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 8, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 9, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 10, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 11, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 12, np.newaxis])
        stage2.append(loss_upsampling22[:, :, :, :, 13, np.newaxis])



        with tf.name_scope('Loss'):
            loss_dic = self.loss_instance.loss_selector('Multistage_ssim_perf_angio_loss',
                                                        labels=labels,logits=logits,
                                                        stage1=stage1,
                                                        stage2=stage2)
            cost = tf.reduce_mean(loss_dic["loss"], name="cost")
            cost_angio = tf.reduce_mean(loss_dic["angio_loss"], name="angio_loss")
            cost_perf = tf.reduce_mean(loss_dic["perf_loss"], name="perf_loss")

        #{'loss': loss, 'loss_s1': loss_s1, 'loss_s2': loss_s2, 'angio_loss': angio_SSIM,         'perf_loss': perf_SSIM + w_s1 * loss_s1 + w_s2 * loss_s2}
        with tf.name_scope('quantifications'):
            [wm_sig, gm_sig, csf_sig] = self.quantifications.seg_sig(segmentation=segmentation, logits=logits)

        with tf.name_scope('ssim_quantifications'):
            [WM_ssim, GM_ssim] = self.loss_instance.ssim_seg(segmentation=segmentation, logits=logits, labels=labels)

        gm_sig_tens = tf.placeholder(tf.float32, name='gm_sig_tens')
        wm_sig_tens = tf.placeholder(tf.float32, name='wm_sig_tens')
        gm_ssim_tens = tf.placeholder(tf.float32, name='gm_ssim_tens')
        wm_ssim_tens = tf.placeholder(tf.float32, name='wm_ssim_tens')

        all_loss = tf.placeholder(tf.float32, name='loss')
        with tf.name_scope('validation'):
            ave_loss = all_loss

        tf.summary.scalar("Loss/loss", cost)
        tf.summary.scalar("Loss/cost_angio", cost_angio)
        tf.summary.scalar("Loss/cost_perf", cost_perf)
        tf.summary.scalar("ave_loss", ave_loss)

        tf.summary.scalar("Quantification/WM_sig", wm_sig)
        tf.summary.scalar("Quantification/GM_sig", gm_sig)

        tf.summary.scalar("ssim_Quantification/WM_ssim", WM_ssim)
        tf.summary.scalar("ssim_Quantification/GM_ssim", GM_ssim)
        # ============================================

        '''AdamOptimizer:'''
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        sess.run(tf.global_variables_initializer())
        logging.debug('total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        summ = tf.summary.merge_all()

        point = 0
        if loadModel:
            chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/MRI_in/experiment-2/unet_checkpoints/'
            ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            point = (ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            point = int(int(point) / self.display_train_step) * self.display_train_step

        '''loop for epochs'''
        for epoch in range(self.total_epochs):
            while self.no_sample_per_each_itr * int(point / self.no_sample_per_each_itr) < self.sample_no:
                print('0')
                print("epoch #: %d" % (epoch))
                startTime = time.time()

                step = 0
                # =============validation================
                if point % self.display_validation_step == 0:
                    '''Validation: '''
                    loss_validation = 0
                    loss_validation_p_ssim=0
                    loss_validation_p_perceptual=0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation = 0

                    wm_sig_val = 0
                    gm_sig_val = 0
                    WM_ssim_val = 0
                    GM_ssim_val = 0


                while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):

                    [crush, noncrush, perf, angio,mri,segmentation_] = _image_class_vl.return_patches_vl(
                        validation_step * self.batch_no_validation,
                        (validation_step + 1) * self.batch_no_validation, is_tr=False
                        )
                    if (len(segmentation_) < self.batch_no_validation):
                        _read_thread_vl.resume()
                        time.sleep(0.5)
                        continue
                    # continue

                    [loss_vali,wm_sig_vl,
                     gm_sig_vl, WM_ssim_vl, GM_ssim_vl] = sess.run([cost, wm_sig, gm_sig, WM_ssim, GM_ssim],
                                  feed_dict={img_row1: crush[:, 0, :, :, :, :],
                                             img_row2: noncrush[:, 1, :, :, :, :],
                                             img_row3: crush[:, 2, :, :, :, :],
                                             img_row4: noncrush[:, 3, :, :, :, :],
                                             img_row5: crush[:, 4, :, :, :, :],
                                             img_row6: noncrush[:, 5, :, :, :, :],
                                             img_row7: crush[:, 6, :, :, :, :],
                                             img_row8: noncrush[:, 7, :, :, :, :],
                                             mri_ph: mri[:, 0, :, :, :, :],
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
                                             segmentation: segmentation_[:, 0, :, :, :, :],
                                             is_training: False,
                                             input_dim: self.patch_window,
                                             all_loss: -1.,
                                             gm_sig_tens: -1,
                                             wm_sig_tens: -1,
                                             gm_ssim_tens: -1,
                                             wm_ssim_tens: -1
                                             })
                    wm_sig_val += wm_sig_vl
                    gm_sig_val += gm_sig_vl
                    WM_ssim_val += WM_ssim_vl
                    GM_ssim_val += GM_ssim_vl
                    loss_validation += loss_vali

                    validation_step += 1
                    if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                        print('nan problem')
                    process = psutil.Process(os.getpid())
                    print(
                        '%d - > %d:   loss_validation: %f, memory_percent: %4s' % (
                            validation_step, validation_step * self.batch_no_validation
                            , loss_vali, str(process.memory_percent()),
                        ))

                settings.queue_isready_vl = False
                acc_validation = acc_validation / (validation_step)
                loss_validation = loss_validation / (validation_step)
                loss_validation_p_ssim = loss_validation_p_ssim / (validation_step)
                loss_validation_p_perceptual = loss_validation_p_perceptual / (validation_step)
                dsc_validation = dsc_validation / (validation_step)

                if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                    print('nan problem')
                _fill_thread_vl.kill_thread()
                print('******Validation, step: %d , accuracy: %.4f, loss: %f*******' % (
                    point, acc_validation, loss_validation))
                [sum_validation] = sess.run([summ],
                                            feed_dict={img_row1: crush[:, 0, :, :, :, :],
                                                       img_row2: noncrush[:, 1, :, :, :, :],
                                                       img_row3: crush[:, 2, :, :, :, :],
                                                       img_row4: noncrush[:, 3, :, :, :, :],
                                                       img_row5: crush[:, 4, :, :, :, :],
                                                       img_row6: noncrush[:, 5, :, :, :, :],
                                                       img_row7: crush[:, 6, :, :, :, :],
                                                       img_row8: noncrush[:, 7, :, :, :, :],
                                                       mri_ph: mri[:, 0, :, :, :, :],
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
                                                       segmentation: segmentation_[:, 0, :, :, :, :],
                                                       is_training: False,
                                                       input_dim: self.patch_window,
                                                       all_loss: loss_validation,
                                                       gm_sig_tens: gm_sig_val,
                                                       wm_sig_tens: wm_sig_val,
                                                       gm_ssim_tens: GM_ssim_val,
                                                       wm_ssim_tens: WM_ssim_val
                                                       })
                validation_writer.add_summary(sum_validation, point)
                print('end of validation---------%d' % (point))

                '''loop for training batches'''
                while (step * self.batch_no < self.no_sample_per_each_itr):

                    # [train_CT_image_patchs, train_GTV_label, train_Penalize_patch,loss_coef_weights] = _image_class.return_patches( self.batch_no)

                    [crush, noncrush, perf, angio, mri, segmentation_] = _image_class.return_patches_tr(self.batch_no)

                    if (len(segmentation_) < self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue

                    if point % self.display_train_step == 0:
                        '''train: '''
                        train_step=0

                    while (train_step <self.display_train_step):
                        if train_step==0:
                            average_loss_train1=0
                            wm_sig_tra = 0
                            gm_sig_tra = 0
                            WM_ssim_tra = 0
                            GM_ssim_tra = 0
                        [loss_train1, optimizing, out,
                         wm_sig_tr, gm_sig_tr, WM_ssim_tr, GM_ssim_tr ] = sess.run([cost, optimizer, y,
                                                                                   wm_sig, gm_sig, WM_ssim, GM_ssim],
                                      feed_dict={img_row1: crush[:, 0, :, :, :, :],
                                                 img_row2: noncrush[:, 1, :, :, :, :],
                                                 img_row3: crush[:, 2, :, :, :, :],
                                                 img_row4: noncrush[:, 3, :, :, :, :],
                                                 img_row5: crush[:, 4, :, :, :, :],
                                                 img_row6: noncrush[:, 5, :, :, :, :],
                                                 img_row7: crush[:, 6, :, :, :, :],
                                                 img_row8: noncrush[:, 7, :, :, :, :],
                                                 mri_ph: mri[:, 0, :, :, :, :],
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
                                                 segmentation: segmentation_[:, 0, :, :, :, :],
                                                 all_loss: -1,
                                                 gm_sig_tens: -1,
                                                 wm_sig_tens: -1,
                                                 gm_ssim_tens: -1,
                                                 wm_ssim_tens: -1
                                                 })
                        average_loss_train1+=loss_train1
                        wm_sig_tra += wm_sig_tr
                        gm_sig_tra += gm_sig_tr
                        WM_ssim_tra += WM_ssim_tr
                        GM_ssim_tra += GM_ssim_tr
                        train_step=train_step+1
                        print(
                            'point: %d, step*self.batch_no:%f , LR: %.15f, loss_train1:%f,memory_percent: %4s, ' % (
                                int((point+train_step)),
                                step * self.batch_no, self.learning_rate, loss_train1,
                                str(process.memory_percent())))

                    average_loss_train1 /= self.display_train_step


                    [sum_train] = sess.run([summ],
                                           feed_dict={img_row1: crush[:, 0, :, :, :, :],
                                                      img_row2: noncrush[:, 1, :, :, :, :],
                                                      img_row3: crush[:, 2, :, :, :, :],
                                                      img_row4: noncrush[:, 3, :, :, :, :],
                                                      img_row5: crush[:, 4, :, :, :, :],
                                                      img_row6: noncrush[:, 5, :, :, :, :],
                                                      img_row7: crush[:, 6, :, :, :, :],
                                                      img_row8: noncrush[:, 7, :, :, :, :],
                                                      mri_ph: mri[:, 0, :, :, :, :],
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
                                                      all_loss: loss_train1,
                                                      segmentation: segmentation_[:, 0, :, :, :, :],
                                                      gm_sig_tens: gm_sig_tra,
                                                      wm_sig_tens: wm_sig_tra,
                                                      gm_ssim_tens: GM_ssim_tra,
                                                      wm_ssim_tens: WM_ssim_tra


                                                      })
                    train_writer.add_summary(sum_train, point)
                    step = step + 1

                    point = point + self.display_train_step
                    if point % 200 == 0:
                        break
                    process = psutil.Process(os.getpid())
                    print(
                        'point: %d, step*self.batch_no:%f , LR: %.15f, loss_train1:%f,memory_percent: %4s' % (
                            int((point)),
                            step * self.batch_no, self.learning_rate, loss_train1,
                            str(process.memory_percent())))
                    point = int((point))  # (self.no_sample_per_each_itr/self.batch_no)*itr1+step

                    if point % 100 == 0:
                        '''saveing model inter epoch'''
                        chckpnt_path = os.path.join(self.chckpnt_dir,
                                                    ('unet_inter_epoch%d_point%d.ckpt' % (epoch, point)))
                        saver.save(sess, chckpnt_path, global_step=point)



            endTime = time.time()
            # ==============end of epoch:
            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'unet.ckpt')
            saver.save(sess, chckpnt_path, global_step=epoch)

            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))
