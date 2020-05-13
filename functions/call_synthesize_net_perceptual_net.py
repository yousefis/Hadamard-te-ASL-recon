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
from functions.network.perceptual_densenet import _densenet
# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread
from functions.test_dir.check_image_range import *
from functions.utils.gradients import gradients
# --------------------------------------------------------------------------------------------------------
class synthesize_net_perceptual_net:
    def __init__(self, data, sample_no, validation_samples, no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag, img_name, label_name, torso_tag, log_tag, min_range, max_range,
                 Logs, fold, Server,newdataset,mixedup=False,l_regu=None):
        settings.init()
        self.mixedup = mixedup
        # ==================================
        self.l_regu=l_regu
        self.train_tag = train_tag
        self.validation_tag = validation_tag
        self.test_tag = test_tag
        self.img_name = img_name
        self.label_name = label_name
        self.torso_tag = torso_tag
        self.data = data
        self.display_train_step = 25
        self.newdataset=newdataset
        # ==================================
        settings.validation_totalimg_patch = validation_samples
        self.gradients=gradients
        # ==================================
        self.learning_decay = .95
        self.learning_rate = 1E-4
        self.beta_rate = 0.05

        self.img_padded_size = 519
        self.seg_size = 505
        self.min_range = min_range
        self.max_range = max_range

        self.label_patchs_size =39#63
        self.patch_window = 53#77#89
        self.sample_no = sample_no
        self.batch_no = 6
        self.batch_no_validation = 6
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance = _loss_func()
        self.fold = fold

        if Server == 'DL':
            self.parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
            if newdataset==True:
                self.data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation00_low/'
            else:
                self.data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01//'
        else:
            self.parent_path = '/exports/lkeb-hpc/syousefi/Code/'
            if newdataset == True:
                self.data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'
            else:
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



    def run_net(self):

        # pre_bn=tf.placeholder(tf.float32,shape=[None,None,None,None,None])
        # image=tf.placeholder(tf.float32,shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window,1])
        # label=tf.placeholder(tf.float32,shape=[self.batch_no_validation,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size,2])
        # loss_coef=tf.placeholder(tf.float32,shape=[self.batch_no_validation,1,1,1])
        # ===================================================================================
        _rd = _read_data(data=self.data,
                         img_name=self.img_name, label_name=self.label_name,
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
                                   fold=self.fold,mixedup=self.mixedup)

        _fill_thread.start()
        patch_extractor_thread.start()

        _read_thread = read_thread(_fill_thread, mutex=settings.mutex, is_training=1)
        _read_thread.start()
        # ===================================================================================
        img_row1 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row1')
        img_row2 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row2')
        img_row3 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row3')
        img_row4 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row4')
        img_row5 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row5')
        img_row6 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row6')
        img_row7 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row7')
        img_row8 = tf.placeholder(tf.float32,
                                  shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                  name='img_row8')

        mri_ph = tf.placeholder(tf.float32,
                                shape=[self.batch_no, self.patch_window, self.patch_window, self.patch_window, 1],
                                name='mri')



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

        # unet=_unet()
        densenet = _densenet()

        [y,_] = densenet.densenet(img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4,
                              img_row5=img_row5,
                              img_row6=img_row6, img_row7=img_row7, img_row8=img_row8, input_dim=input_dim,
                              is_training=is_training)
        self.vgg = vgg_feature_maker()
        # self.vgg_y0 = self.vgg.feed_img(loss_placeholder)

        feature_type='huber'
        self.vgg_y0 = self.vgg.feed_img(y[:, :, :, :, 0],feature_type=feature_type).copy()
        self.vgg_y1 = self.vgg.feed_img(y[:, :, :, :, 1],feature_type=feature_type).copy()
        self.vgg_y2 = self.vgg.feed_img(y[:, :, :, :, 2],feature_type=feature_type).copy()
        self.vgg_y3 = self.vgg.feed_img(y[:, :, :, :, 3],feature_type=feature_type).copy()
        self.vgg_y4 = self.vgg.feed_img(y[:, :, :, :, 4],feature_type=feature_type).copy()
        self.vgg_y5 = self.vgg.feed_img(y[:, :, :, :, 5],feature_type=feature_type).copy()
        self.vgg_y6 = self.vgg.feed_img(y[:, :, :, :, 6],feature_type=feature_type).copy()


        self.vgg_y7 = self.vgg.feed_img(y[:, :, :, :, 7],feature_type=feature_type)
        self.vgg_y8 = self.vgg.feed_img(y[:, :, :, :, 8],feature_type=feature_type)
        self.vgg_y9 = self.vgg.feed_img(y[:, :, :, :, 9],feature_type=feature_type)
        self.vgg_y10 = self.vgg.feed_img(y[:, :, :, :, 10],feature_type=feature_type)
        self.vgg_y11 = self.vgg.feed_img(y[:, :, :, :, 11],feature_type=feature_type)
        self.vgg_y12 = self.vgg.feed_img(y[:, :, :, :, 12],feature_type=feature_type)
        self.vgg_y13 = self.vgg.feed_img(y[:, :, :, :, 13],feature_type=feature_type)

        self.vgg_label0 = self.vgg.feed_img(label1[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label1 = self.vgg.feed_img(label2[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label2 = self.vgg.feed_img(label3[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label3 = self.vgg.feed_img(label4[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label4 = self.vgg.feed_img(label5[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label5 = self.vgg.feed_img(label6[:,:,:,:,0],feature_type=feature_type).copy()
        self.vgg_label6 = self.vgg.feed_img(label7[:,:,:,:,0],feature_type=feature_type).copy()


        self.vgg_label7 = self.vgg.feed_img(label8[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label8 = self.vgg.feed_img(label9[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label9 = self.vgg.feed_img(label10[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label10 = self.vgg.feed_img(label11[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label11 = self.vgg.feed_img(label12[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label12 = self.vgg.feed_img(label13[:,:,:,:,0],feature_type=feature_type)
        self.vgg_label13 = self.vgg.feed_img(label14[:,:,:,:,0],feature_type=feature_type)

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
        # run_opts = tf.RunOptions(report_tensor_allocations_upon_oom=True)

        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' , graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' , graph=sess.graph)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000)

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

        vgg_in_feature=[]
        vgg_in_feature.append(self.vgg_y0)
        vgg_in_feature.append(self.vgg_y1)
        vgg_in_feature.append(self.vgg_y2)
        vgg_in_feature.append(self.vgg_y3)
        vgg_in_feature.append(self.vgg_y4)
        vgg_in_feature.append(self.vgg_y5)
        vgg_in_feature.append(self.vgg_y6)

        vgg_in_feature.append(self.vgg_y7)
        vgg_in_feature.append(self.vgg_y8)
        vgg_in_feature.append(self.vgg_y9)
        vgg_in_feature.append(self.vgg_y10)
        vgg_in_feature.append(self.vgg_y11)
        vgg_in_feature.append(self.vgg_y12)
        vgg_in_feature.append(self.vgg_y13)

        vgg_label_feature=[]
        vgg_label_feature.append(self.vgg_label0)
        vgg_label_feature.append(self.vgg_label1)
        vgg_label_feature.append(self.vgg_label2)
        vgg_label_feature.append(self.vgg_label3)
        vgg_label_feature.append(self.vgg_label4)
        vgg_label_feature.append(self.vgg_label5)
        vgg_label_feature.append(self.vgg_label6)

        vgg_label_feature.append(self.vgg_label7)
        vgg_label_feature.append(self.vgg_label8)
        vgg_label_feature.append(self.vgg_label9)
        vgg_label_feature.append(self.vgg_label10)
        vgg_label_feature.append(self.vgg_label11)
        vgg_label_feature.append(self.vgg_label12)
        vgg_label_feature.append(self.vgg_label13)

        l_regu=self.l_regu#'l2_regularizer'
        '''AdamOptimizer:'''
        with tf.name_scope('Loss'):
            loss_dic = self.loss_instance.loss_selector('content_vgg_pairwise_loss_huber',
                                                        labels=vgg_label_feature, logits=vgg_in_feature, vgg=self.vgg,
                                                        h_labels=labels, h_logits =logits)

            cost = tf.reduce_sum(loss_dic['loss'], name="cost")
            # ============================================
            regularization_penalty=0
            if l_regu=='l1_regularizer':
                l1_regularizer = tf.contrib.layers.l1_regularizer(
                    scale=0.005, scope=None
                )
                weights = tf.trainable_variables()  # all vars of your graph
                regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            elif l_regu=='l2_regularizer':
                l2_regularizer = tf.contrib.layers.l2_regularizer(
                    scale=0.001, scope=None
                )
                weights = tf.trainable_variables()  # all vars of your graph
                regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
            cost=cost+regularization_penalty

            angio_loss = tf.reduce_sum(loss_dic['angio_loss'], name="angio_loss")
            perf_loss = tf.reduce_sum(loss_dic['perf_loss'], name="perf_loss")

            # loss_dic = {'loss': vgg_loss + huber_loss,
            #             'vgg_loss': vgg_loss,  # 1
            #             'vgg_perf': vgg_perf,  # 1
            #             'vgg_angio': vgg_angio,  # 1
            #             'vgg_losses': vgg_losses,  # 14 all losses
            #             'huber_loss': huber_loss,  # 1
            #             'huber_perf': huber_perf,  # 7
            #             'huber_angio': huber_angio,  # 7
            #             }

            vgg_losses = loss_dic['vgg_losses']
            perf_huber=loss_dic['huber_perf']
            angio_huber=loss_dic['huber_angio']
            huber_loss=loss_dic['huber_loss']
            perceptual_loss=loss_dic['vgg_loss']
        # ============================================
        #average vgg_losses vgg
        perf_vgg_loss_tens = tf.placeholder(tf.float32, name='VGG_perf')
        angio_vgg_loss_tens = tf.placeholder(tf.float32, name='VGG_angio')
        with tf.variable_scope("1_VGG"):
            tf.summary.scalar("1_VGG/perf_vgg_loss_tens", perf_vgg_loss_tens)
            tf.summary.scalar("1_VGG/angio_vgg_loss_tens", angio_vgg_loss_tens)
        # ============================================
        # average vgg_losses huber
        perf_huber_loss_tens = tf.placeholder(tf.float32, name='huber_perf')
        angio_huber_loss_tens = tf.placeholder(tf.float32, name='huber_angio')
        with tf.variable_scope("2_Huber"):
            tf.summary.scalar("2_Huber/perf_huber_loss_tens", perf_huber_loss_tens)
            tf.summary.scalar("2_Huber/angio_huber_loss_tens", angio_huber_loss_tens)
        # ============================================
        with tf.variable_scope("0_Loss"):
            tf.summary.scalar("0_Loss/Loss", loss_dic['loss'])
        # ============================================
        perf_vgg_tens0 = tf.placeholder(tf.float32, name='vgg_perf0')
        perf_vgg_tens1 = tf.placeholder(tf.float32, name='vgg_perf1')
        perf_vgg_tens2 = tf.placeholder(tf.float32, name='vgg_perf2')
        perf_vgg_tens3 = tf.placeholder(tf.float32, name='vgg_perf3')
        perf_vgg_tens4 = tf.placeholder(tf.float32, name='vgg_perf4')
        perf_vgg_tens5 = tf.placeholder(tf.float32, name='vgg_perf5')
        perf_vgg_tens6 = tf.placeholder(tf.float32, name='vgg_perf6')

        angio_vgg_tens0 = tf.placeholder(tf.float32, name='vgg_angio0')
        angio_vgg_tens1 = tf.placeholder(tf.float32, name='vgg_angio1')
        angio_vgg_tens2 = tf.placeholder(tf.float32, name='vgg_angio2')
        angio_vgg_tens3 = tf.placeholder(tf.float32, name='vgg_angio3')
        angio_vgg_tens4 = tf.placeholder(tf.float32, name='vgg_angio4')
        angio_vgg_tens5 = tf.placeholder(tf.float32, name='vgg_angio5')
        angio_vgg_tens6 = tf.placeholder(tf.float32, name='vgg_angio6')
        # ============================================

        # perceptual_loss_tens = tf.placeholder(tf.float32, name='perceptual_loss')

        perf_huber_tens0 = tf.placeholder(tf.float32, name='huber_perf0')
        perf_huber_tens1 = tf.placeholder(tf.float32, name='huber_perf1')
        perf_huber_tens2 = tf.placeholder(tf.float32, name='huber_perf2')
        perf_huber_tens3 = tf.placeholder(tf.float32, name='huber_perf3')
        perf_huber_tens4 = tf.placeholder(tf.float32, name='huber_perf4')
        perf_huber_tens5 = tf.placeholder(tf.float32, name='huber_perf5')
        perf_huber_tens6 = tf.placeholder(tf.float32, name='huber_perf6')

        angio_huber_tens0 = tf.placeholder(tf.float32, name='huber_angio0')
        angio_huber_tens1 = tf.placeholder(tf.float32, name='huber_angio1')
        angio_huber_tens2 = tf.placeholder(tf.float32, name='huber_angio2')
        angio_huber_tens3 = tf.placeholder(tf.float32, name='huber_angio3')
        angio_huber_tens4 = tf.placeholder(tf.float32, name='huber_angio4')
        angio_huber_tens5 = tf.placeholder(tf.float32, name='huber_angio5')
        angio_huber_tens6 = tf.placeholder(tf.float32, name='huber_angio6')
        
        #============================================
        with tf.variable_scope("3_VGG_perf"):
            tf.summary.scalar("3_VGG_perf/perfusion0", perf_vgg_tens0)#vgg_losses[0])
            tf.summary.scalar("3_VGG_perf/perfusion1", perf_vgg_tens1)#vgg_losses[1])
            tf.summary.scalar("3_VGG_perf/perfusion2", perf_vgg_tens2)#vgg_losses[2])
            tf.summary.scalar("3_VGG_perf/perfusion3", perf_vgg_tens3)#vgg_losses[3])
            tf.summary.scalar("3_VGG_perf/perfusion4", perf_vgg_tens4)#vgg_losses[4])
            tf.summary.scalar("3_VGG_perf/perfusion5", perf_vgg_tens5)#vgg_losses[5])
            tf.summary.scalar("3_VGG_perf/perfusion6", perf_vgg_tens6)#vgg_losses[6])
        with tf.variable_scope("4_VGG_angio"):
            tf.summary.scalar("4_VGG_angio/angio0", angio_vgg_tens0)#vgg_losses[7])
            tf.summary.scalar("4_VGG_angio/angio1", angio_vgg_tens1)#vgg_losses[8])
            tf.summary.scalar("4_VGG_angio/angio2", angio_vgg_tens2)#vgg_losses[9])
            tf.summary.scalar("4_VGG_angio/angio3", angio_vgg_tens3)#vgg_losses[10])
            tf.summary.scalar("4_VGG_angio/angio4", angio_vgg_tens4)#vgg_losses[11])
            tf.summary.scalar("4_VGG_angio/angio5", angio_vgg_tens5)#vgg_losses[12])
            tf.summary.scalar("4_VGG_angio/angio6", angio_vgg_tens6)#vgg_losses[13])
        #============================================
        with tf.variable_scope("5_Huber_perf"):
            tf.summary.scalar("5_Huber_perf/perfusion0", perf_huber_tens0)#perf_huber[0])
            tf.summary.scalar("5_Huber_perf/perfusion1", perf_huber_tens1)#perf_huber[1])
            tf.summary.scalar("5_Huber_perf/perfusion2", perf_huber_tens2)#perf_huber[2])
            tf.summary.scalar("5_Huber_perf/perfusion3", perf_huber_tens3)#perf_huber[3])
            tf.summary.scalar("5_Huber_perf/perfusion4", perf_huber_tens4)#perf_huber[4])
            tf.summary.scalar("5_Huber_perf/perfusion5", perf_huber_tens5)#perf_huber[5])
            tf.summary.scalar("5_Huber_perf/perfusion6", perf_huber_tens6)#perf_huber[6])
        with tf.variable_scope("6_Huber_angio"):
            tf.summary.scalar("6_Huber_angio/angio0", angio_huber_tens0)#angio_huber[0])
            tf.summary.scalar("6_Huber_angio/angio1", angio_huber_tens1)#angio_huber[1])
            tf.summary.scalar("6_Huber_angio/angio2", angio_huber_tens2)#angio_huber[2])
            tf.summary.scalar("6_Huber_angio/angio3", angio_huber_tens3)#angio_huber[3])
            tf.summary.scalar("6_Huber_angio/angio4", angio_huber_tens4)#angio_huber[4])
            tf.summary.scalar("6_Huber_angio/angio5", angio_huber_tens5)#angio_huber[5])
            tf.summary.scalar("6_Huber_angio/angio6", angio_huber_tens6)#angio_huber[6])

        
        all_loss = tf.placeholder(tf.float32, name='loss')

        with tf.name_scope('validation'):
            ave_loss = all_loss
        with tf.variable_scope("ave_loss"):
            tf.summary.scalar("ave_loss", ave_loss)

        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        sess.run(tf.global_variables_initializer())
        logging.debug('total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))

        summ = tf.summary.merge_all()
        loadModel = 1
        point = 666200
        if loadModel:
            chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/Log_perceptual/regularization/perceptual-0/'
            ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
            point = np.int16(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])



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
                    loss_validation_angio_vl=0
                    loss_validation_perf_vl=0
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation = 0
                    vgg_loss_p0_vl=0
                    vgg_loss_p1_vl=0
                    vgg_loss_p2_vl=0
                    vgg_loss_p3_vl=0
                    vgg_loss_p4_vl=0
                    vgg_loss_p5_vl=0
                    vgg_loss_p6_vl=0

                    vgg_loss_a0_vl=0
                    vgg_loss_a1_vl=0
                    vgg_loss_a2_vl=0
                    vgg_loss_a3_vl=0
                    vgg_loss_a4_vl=0
                    vgg_loss_a5_vl=0
                    vgg_loss_a6_vl=0

                    huber_loss_p0_vl = 0
                    huber_loss_p1_vl = 0
                    huber_loss_p2_vl = 0
                    huber_loss_p3_vl = 0
                    huber_loss_p4_vl = 0
                    huber_loss_p5_vl = 0
                    huber_loss_p6_vl = 0

                    huber_loss_a0_vl = 0
                    huber_loss_a1_vl = 0
                    huber_loss_a2_vl = 0
                    huber_loss_a3_vl = 0
                    huber_loss_a4_vl = 0
                    huber_loss_a5_vl = 0
                    huber_loss_a6_vl = 0

                    huber_perf_vl=0
                    huber_angio_vl=0


                while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):

                    [crush, noncrush, perf, angio,mri,segmentation] = _image_class_vl.return_patches_vl(
                        validation_step * self.batch_no_validation,
                        (validation_step + 1) * self.batch_no_validation, is_tr=False
                        )
                    if (len(segmentation) < self.batch_no_validation):
                        _read_thread_vl.resume()
                        time.sleep(0.5)
                        continue
                    # continue

                    [loss_vali,perf_loss_vl,angio_loss_vl,vgg_losses_vl,
                     perf_huber_vl,angio_huber_vl,huber_loss_vl] = sess.run([cost,perf_loss,angio_loss,vgg_losses,
                                                               perf_huber,angio_huber,huber_loss],
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
                                             all_loss: -1, #whole loss

                                             angio_vgg_loss_tens:-1, #vgg angio
                                             perf_vgg_loss_tens:-1,

                                             perf_vgg_tens0:-1,
                                             perf_vgg_tens1:-1,
                                             perf_vgg_tens2:-1,
                                             perf_vgg_tens3:-1,
                                             perf_vgg_tens4:-1,
                                             perf_vgg_tens5:-1,
                                             perf_vgg_tens6:-1,
                                             angio_vgg_tens0:-1,
                                             angio_vgg_tens1:-1,
                                             angio_vgg_tens2:-1,
                                             angio_vgg_tens3:-1,
                                             angio_vgg_tens4:-1,
                                             angio_vgg_tens5:-1,
                                             angio_vgg_tens6:-1,

                                             perf_huber_loss_tens:-1,
                                             angio_huber_loss_tens:-1,

                                             perf_huber_tens0:-1,
                                             perf_huber_tens1:-1,
                                             perf_huber_tens2:-1,
                                             perf_huber_tens3:-1,
                                             perf_huber_tens4:-1,
                                             perf_huber_tens5:-1,
                                             perf_huber_tens6:-1,

                                             angio_huber_tens0:-1,
                                             angio_huber_tens1:-1,
                                             angio_huber_tens2:-1,
                                             angio_huber_tens3:-1,
                                             angio_huber_tens4:-1,
                                             angio_huber_tens5:-1,
                                             angio_huber_tens6:-1,
                                             })
                    loss_validation += loss_vali
                    loss_validation_perf_vl += perf_loss_vl
                    loss_validation_angio_vl += angio_loss_vl

                    vgg_loss_p0_vl+=vgg_losses_vl[0]
                    vgg_loss_p1_vl+=vgg_losses_vl[1]
                    vgg_loss_p2_vl+=vgg_losses_vl[2]
                    vgg_loss_p3_vl+=vgg_losses_vl[3]
                    vgg_loss_p4_vl+=vgg_losses_vl[4]
                    vgg_loss_p5_vl+=vgg_losses_vl[5]
                    vgg_loss_p6_vl+=vgg_losses_vl[6]

                    vgg_loss_a0_vl+=vgg_losses_vl[0]
                    vgg_loss_a1_vl+=vgg_losses_vl[1]
                    vgg_loss_a2_vl+=vgg_losses_vl[2]
                    vgg_loss_a3_vl+=vgg_losses_vl[3]
                    vgg_loss_a4_vl+=vgg_losses_vl[4]
                    vgg_loss_a5_vl+=vgg_losses_vl[5]
                    vgg_loss_a6_vl+=vgg_losses_vl[6]

                    huber_loss_p0_vl += perf_huber_vl[0]
                    huber_loss_p1_vl += perf_huber_vl[1]
                    huber_loss_p2_vl += perf_huber_vl[2]
                    huber_loss_p3_vl += perf_huber_vl[3]
                    huber_loss_p4_vl += perf_huber_vl[4]
                    huber_loss_p5_vl += perf_huber_vl[5]
                    huber_loss_p6_vl += perf_huber_vl[6]

                    huber_loss_a0_vl += angio_huber_vl[0]
                    huber_loss_a1_vl += angio_huber_vl[1]
                    huber_loss_a2_vl += angio_huber_vl[2]
                    huber_loss_a3_vl += angio_huber_vl[3]
                    huber_loss_a4_vl += angio_huber_vl[4]
                    huber_loss_a5_vl += angio_huber_vl[5]
                    huber_loss_a6_vl += angio_huber_vl[6]

                    validation_step += 1
                    if np.isnan(dsc_validation) or np.isnan(loss_validation) or np.isnan(acc_validation):
                        print('nan problem')
                    process = psutil.Process(os.getpid())
                    print(
                        '%d - > %d:   loss_validation: %f, memory_percent: %4s' % (
                            validation_step, validation_step * self.batch_no_validation
                            , loss_vali, str(process.memory_percent())
                        ))

                settings.queue_isready_vl = False
                acc_validation = acc_validation / (validation_step)
                loss_validation = loss_validation / (validation_step)
                loss_validation_perf_vl = loss_validation_perf_vl / (validation_step)
                loss_validation_angio_vl = loss_validation_angio_vl / (validation_step)
                dsc_validation = dsc_validation / (validation_step)

                vgg_loss_p0_vl=vgg_loss_p0_vl/ validation_step
                vgg_loss_p1_vl=vgg_loss_p1_vl/ validation_step
                vgg_loss_p2_vl=vgg_loss_p2_vl/ validation_step
                vgg_loss_p3_vl=vgg_loss_p3_vl/ validation_step
                vgg_loss_p4_vl=vgg_loss_p4_vl/ validation_step
                vgg_loss_p5_vl=vgg_loss_p5_vl/ validation_step
                vgg_loss_p6_vl=vgg_loss_p6_vl/ validation_step

                vgg_loss_a0_vl=vgg_loss_a0_vl/ validation_step
                vgg_loss_a1_vl=vgg_loss_a1_vl/ validation_step
                vgg_loss_a2_vl=vgg_loss_a2_vl/ validation_step
                vgg_loss_a3_vl=vgg_loss_a3_vl/ validation_step
                vgg_loss_a4_vl=vgg_loss_a4_vl/ validation_step
                vgg_loss_a5_vl=vgg_loss_a5_vl/ validation_step
                vgg_loss_a6_vl=vgg_loss_a6_vl/ validation_step

                huber_loss_p0_vl=huber_loss_p0_vl/validation_step
                huber_loss_p1_vl=huber_loss_p1_vl/validation_step
                huber_loss_p2_vl=huber_loss_p2_vl/validation_step
                huber_loss_p3_vl=huber_loss_p3_vl/validation_step
                huber_loss_p4_vl=huber_loss_p4_vl/validation_step
                huber_loss_p5_vl=huber_loss_p5_vl/validation_step
                huber_loss_p6_vl=huber_loss_p6_vl/validation_step

                huber_loss_a0_vl=huber_loss_a0_vl/validation_step
                huber_loss_a1_vl=huber_loss_a1_vl/validation_step
                huber_loss_a2_vl=huber_loss_a2_vl/validation_step
                huber_loss_a3_vl=huber_loss_a3_vl/validation_step
                huber_loss_a4_vl=huber_loss_a4_vl/validation_step
                huber_loss_a5_vl=huber_loss_a5_vl/validation_step
                huber_loss_a6_vl=huber_loss_a6_vl/validation_step

                huber_perf_vl=np.sum([huber_loss_p0_vl,huber_loss_p1_vl,huber_loss_p2_vl,huber_loss_p3_vl,huber_loss_p4_vl,huber_loss_p5_vl,huber_loss_p6_vl])
                huber_angio_vl=np.sum([huber_loss_a0_vl,huber_loss_a1_vl,huber_loss_a2_vl,huber_loss_a3_vl,huber_loss_a4_vl,huber_loss_a5_vl,huber_loss_a6_vl])


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
                                                       is_training: False,
                                                       input_dim: self.patch_window,
                                                       all_loss: loss_validation,
                                                       perf_vgg_loss_tens: loss_validation_perf_vl,
                                                       angio_vgg_loss_tens: loss_validation_angio_vl,
                                                       perf_vgg_tens0: vgg_loss_p0_vl,
                                                       perf_vgg_tens1: vgg_loss_p1_vl,
                                                       perf_vgg_tens2: vgg_loss_p2_vl,
                                                       perf_vgg_tens3: vgg_loss_p3_vl,
                                                       perf_vgg_tens4: vgg_loss_p4_vl,
                                                       perf_vgg_tens5: vgg_loss_p5_vl,
                                                       perf_vgg_tens6: vgg_loss_p6_vl,
                                                       angio_vgg_tens0: vgg_loss_a0_vl,
                                                       angio_vgg_tens1: vgg_loss_a1_vl,
                                                       angio_vgg_tens2: vgg_loss_a2_vl,
                                                       angio_vgg_tens3: vgg_loss_a3_vl,
                                                       angio_vgg_tens4: vgg_loss_a4_vl,
                                                       angio_vgg_tens5: vgg_loss_a5_vl,
                                                       angio_vgg_tens6: vgg_loss_a6_vl,

                                                       perf_huber_loss_tens: huber_perf_vl,
                                                       angio_huber_loss_tens: huber_angio_vl,

                                                       perf_huber_tens0: huber_loss_p0_vl,
                                                       perf_huber_tens1: huber_loss_p1_vl,
                                                       perf_huber_tens2: huber_loss_p2_vl,
                                                       perf_huber_tens3: huber_loss_p3_vl,
                                                       perf_huber_tens4: huber_loss_p4_vl,
                                                       perf_huber_tens5: huber_loss_p5_vl,
                                                       perf_huber_tens6: huber_loss_p6_vl,

                                                       angio_huber_tens0: huber_loss_a0_vl,
                                                       angio_huber_tens1: huber_loss_a1_vl,
                                                       angio_huber_tens2: huber_loss_a2_vl,
                                                       angio_huber_tens3: huber_loss_a3_vl,
                                                       angio_huber_tens4: huber_loss_a4_vl,
                                                       angio_huber_tens5: huber_loss_a5_vl,
                                                       angio_huber_tens6: huber_loss_a6_vl,
                                                       })
                validation_writer.add_summary(sum_validation, point)
                print('end of validation---------%d' % (point))

                '''loop for training batches'''
                while (step * self.batch_no < self.no_sample_per_each_itr):

                    # [train_CT_image_patchs, train_GTV_label, train_Penalize_patch,loss_coef_weights] = _image_class.return_patches( self.batch_no)

                    [crush, noncrush, perf, angio,mri,segmentation] = _image_class.return_patches_tr(self.batch_no)

                    if (len(segmentation) < self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue

                    if point % self.display_train_step == 0:
                        '''train: '''
                        train_step=0

                    while (train_step <self.display_train_step):
                        if train_step==0:
                            average_loss_train1=0
                            loss_validation_angio_tr = 0
                            loss_validation_perf_tr = 0

                            vgg_loss_p0_tr = 0
                            vgg_loss_p1_tr = 0
                            vgg_loss_p2_tr = 0
                            vgg_loss_p3_tr = 0
                            vgg_loss_p4_tr = 0
                            vgg_loss_p5_tr = 0
                            vgg_loss_p6_tr = 0

                            vgg_loss_a0_tr = 0
                            vgg_loss_a1_tr = 0
                            vgg_loss_a2_tr = 0
                            vgg_loss_a3_tr = 0
                            vgg_loss_a4_tr = 0
                            vgg_loss_a5_tr = 0
                            vgg_loss_a6_tr = 0

                            huber_loss_p0_tr = 0
                            huber_loss_p1_tr = 0
                            huber_loss_p2_tr = 0
                            huber_loss_p3_tr = 0
                            huber_loss_p4_tr = 0
                            huber_loss_p5_tr = 0
                            huber_loss_p6_tr = 0

                            huber_loss_a0_tr = 0
                            huber_loss_a1_tr = 0
                            huber_loss_a2_tr = 0
                            huber_loss_a3_tr = 0
                            huber_loss_a4_tr = 0
                            huber_loss_a5_tr = 0
                            huber_loss_a6_tr = 0
                        [loss_train1, optimizing, out,
                         perf_loss_tr, angio_loss_tr ,losses_tr,
                         perf_huber_tr, angio_huber_tr, huber_loss_tr] = sess.run([cost, optimizer, y,perf_loss,angio_loss,vgg_losses,
                                                                                   perf_huber, angio_huber, huber_loss],
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
                                                 all_loss: -1,
                                                 angio_vgg_loss_tens: -1,
                                                 perf_vgg_loss_tens: -1,
                                                 perf_vgg_tens0: -1,
                                                 perf_vgg_tens1: -1,
                                                 perf_vgg_tens2: -1,
                                                 perf_vgg_tens3: -1,
                                                 perf_vgg_tens4: -1,
                                                 perf_vgg_tens5: -1,
                                                 perf_vgg_tens6: -1,
                                                 angio_vgg_tens0: -1,
                                                 angio_vgg_tens1: -1,
                                                 angio_vgg_tens2: -1,
                                                 angio_vgg_tens3: -1,
                                                 angio_vgg_tens4: -1,
                                                 angio_vgg_tens5: -1,
                                                 angio_vgg_tens6: -1,
                                                 perf_huber_loss_tens: -1,
                                                 angio_huber_loss_tens: -1,
                                                 perf_huber_tens0: -1,
                                                 perf_huber_tens1: -1,
                                                 perf_huber_tens2: -1,
                                                 perf_huber_tens3: -1,
                                                 perf_huber_tens4: -1,
                                                 perf_huber_tens5: -1,
                                                 perf_huber_tens6: -1,
                                                 angio_huber_tens0: -1,
                                                 angio_huber_tens1: -1,
                                                 angio_huber_tens2: -1,
                                                 angio_huber_tens3: -1,
                                                 angio_huber_tens4: -1,
                                                 angio_huber_tens5: -1,
                                                 angio_huber_tens6: -1,
                                                 })
                        average_loss_train1+=loss_train1
                        loss_validation_angio_tr+=angio_loss_tr
                        loss_validation_perf_tr+=perf_loss_tr

                        vgg_loss_p0_tr += losses_tr[0]
                        vgg_loss_p1_tr += losses_tr[1]
                        vgg_loss_p2_tr += losses_tr[2]
                        vgg_loss_p3_tr += losses_tr[3]
                        vgg_loss_p4_tr += losses_tr[4]
                        vgg_loss_p5_tr += losses_tr[5]
                        vgg_loss_p6_tr += losses_tr[6]
                        vgg_loss_a0_tr += losses_tr[0]
                        vgg_loss_a1_tr += losses_tr[1]
                        vgg_loss_a2_tr += losses_tr[2]
                        vgg_loss_a3_tr += losses_tr[3]
                        vgg_loss_a4_tr += losses_tr[4]
                        vgg_loss_a5_tr += losses_tr[5]
                        vgg_loss_a6_tr += losses_tr[6]

                        huber_loss_p0_tr += perf_huber_tr[0]
                        huber_loss_p1_tr += perf_huber_tr[1]
                        huber_loss_p2_tr += perf_huber_tr[2]
                        huber_loss_p3_tr += perf_huber_tr[3]
                        huber_loss_p4_tr += perf_huber_tr[4]
                        huber_loss_p5_tr += perf_huber_tr[5]
                        huber_loss_p6_tr += perf_huber_tr[6]

                        huber_loss_a0_tr += angio_huber_tr[0]
                        huber_loss_a1_tr += angio_huber_tr[1]
                        huber_loss_a2_tr += angio_huber_tr[2]
                        huber_loss_a3_tr += angio_huber_tr[3]
                        huber_loss_a4_tr += angio_huber_tr[4]
                        huber_loss_a5_tr += angio_huber_tr[5]
                        huber_loss_a6_tr += angio_huber_tr[6]
                        train_step = train_step + 1
                        print(
                            'point: %d, step*self.batch_no:%f , LR: %.15f, loss_train1:%f,memory_percent: %4s' % (
                                int((point + train_step)),
                                step * self.batch_no, self.learning_rate, loss_train1,
                                str(process.memory_percent())))
                    average_loss_train1 /= self.display_train_step
                    loss_validation_angio_tr /= self.display_train_step
                    loss_validation_perf_tr /= self.display_train_step

                    vgg_loss_p0_tr = vgg_loss_p0_tr / validation_step
                    vgg_loss_p1_tr = vgg_loss_p1_tr / validation_step
                    vgg_loss_p2_tr = vgg_loss_p2_tr / validation_step
                    vgg_loss_p3_tr = vgg_loss_p3_tr / validation_step
                    vgg_loss_p4_tr = vgg_loss_p4_tr / validation_step
                    vgg_loss_p5_tr = vgg_loss_p5_tr / validation_step
                    vgg_loss_p6_tr = vgg_loss_p6_tr / validation_step

                    vgg_loss_a0_tr = vgg_loss_a0_tr / validation_step
                    vgg_loss_a1_tr = vgg_loss_a1_tr / validation_step
                    vgg_loss_a2_tr = vgg_loss_a2_tr / validation_step
                    vgg_loss_a3_tr = vgg_loss_a3_tr / validation_step
                    vgg_loss_a4_tr = vgg_loss_a4_tr / validation_step
                    vgg_loss_a5_tr = vgg_loss_a5_tr / validation_step
                    vgg_loss_a6_tr = vgg_loss_a6_tr / validation_step

                    huber_perf_tr = np.sum(
                        [huber_loss_p0_tr, huber_loss_p1_tr, huber_loss_p2_tr, huber_loss_p3_tr, huber_loss_p4_tr, huber_loss_p5_tr,
                         huber_loss_p6_tr])
                    huber_angio_tr = np.sum(
                        [huber_loss_a0_tr, huber_loss_a1_tr, huber_loss_a2_tr, huber_loss_a3_tr, huber_loss_a4_tr, huber_loss_a5_tr,
                         huber_loss_a6_tr])

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
                                                      perf_vgg_loss_tens: loss_validation_perf_tr,
                                                      angio_vgg_loss_tens: loss_validation_angio_tr,
                                                      perf_vgg_tens0: vgg_loss_p0_tr,
                                                      perf_vgg_tens1: vgg_loss_p1_tr,
                                                      perf_vgg_tens2: vgg_loss_p2_tr,
                                                      perf_vgg_tens3: vgg_loss_p3_tr,
                                                      perf_vgg_tens4: vgg_loss_p4_tr,
                                                      perf_vgg_tens5: vgg_loss_p5_tr,
                                                      perf_vgg_tens6: vgg_loss_p6_tr,
                                                      angio_vgg_tens0: vgg_loss_a0_tr,
                                                      angio_vgg_tens1: vgg_loss_a1_tr,
                                                      angio_vgg_tens2: vgg_loss_a2_tr,
                                                      angio_vgg_tens3: vgg_loss_a3_tr,
                                                      angio_vgg_tens4: vgg_loss_a4_tr,
                                                      angio_vgg_tens5: vgg_loss_a5_tr,
                                                      angio_vgg_tens6: vgg_loss_a6_tr,
                                                      perf_huber_loss_tens: huber_perf_tr,
                                                      angio_huber_loss_tens: huber_angio_tr,
                                                      perf_huber_tens0: huber_loss_p0_tr,
                                                      perf_huber_tens1: huber_loss_p1_tr,
                                                      perf_huber_tens2: huber_loss_p2_tr,
                                                      perf_huber_tens3: huber_loss_p3_tr,
                                                      perf_huber_tens4: huber_loss_p4_tr,
                                                      perf_huber_tens5: huber_loss_p5_tr,
                                                      perf_huber_tens6: huber_loss_p6_tr,

                                                      angio_huber_tens0: huber_loss_a0_tr,
                                                      angio_huber_tens1: huber_loss_a1_tr,
                                                      angio_huber_tens2: huber_loss_a2_tr,
                                                      angio_huber_tens3: huber_loss_a3_tr,
                                                      angio_huber_tens4: huber_loss_a4_tr,
                                                      angio_huber_tens5: huber_loss_a5_tr,
                                                      angio_huber_tens6: huber_loss_a6_tr,

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
                    point = int((point))

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
