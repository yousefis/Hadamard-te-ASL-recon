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
from functions.network.densenet import _densenet
# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread


# --------------------------------------------------------------------------------------------------------
class synthesize_net:
    def __init__(self,data,  sample_no,validation_samples,no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag,img_name,label_name,torso_tag,log_tag,min_range,max_range,
                 Logs,fold):
        settings.init()
        # ==================================
        self.train_tag=train_tag
        self.validation_tag=validation_tag
        self.test_tag=test_tag
        self.img_name=img_name
        self.label_name=label_name
        self.torso_tag=torso_tag
        self.data=data
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

        self.label_patchs_size =39
        self.patch_window = 53#89
        self.sample_no = sample_no
        self.batch_no =9
        self.batch_no_validation = 9
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance=_loss_func()

        self.parent_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
        self.data_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/'
        self.Logs=Logs

        self.no_sample_per_each_itr=no_sample_per_each_itr


        self.log_ext = log_tag
        self.LOGDIR = self.parent_path+self.Logs + self.log_ext + '/'
        self.chckpnt_dir = self.parent_path+self.Logs + self.log_ext + '/unet_checkpoints/'
        self.x_hist=0

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
        train_data, validation_data, test_data=_rd.read_data_path(fold=self.fold)

        # ======================================
        bunch_of_images_no=20
        sample_no=60
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
        sample_no=60
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

        # img_row1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        # img_row8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1])
        #
        # label1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label9 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label10 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label11 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label12 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label13 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])
        # label14 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1])

        img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])

        label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])


        is_training = tf.placeholder(tf.bool, name='is_training')
        input_dim = tf.placeholder(tf.int32, name='input_dim')


        # unet=_unet()
        densenet=_densenet()

        y=densenet.densenet( img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4, img_row5=img_row5,
                     img_row6=img_row6, img_row7=img_row7, img_row8=img_row8,input_dim=input_dim,is_training=is_training)

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



        show_img=tf.nn.softmax(y)[:, int(self.label_patchs_size / 2) , :, :, 0, np.newaxis]
        tf.summary.image('net_label',show_img  , 3)
        tf.summary.image('out0',y_dirX ,3)
        tf.summary.image('out1',y_dirX1 ,3)
        tf.summary.image('out2',y_dirX2 ,3)
        tf.summary.image('out3',y_dirX3 ,3)
        tf.summary.image('out4',y_dirX4 ,3)
        tf.summary.image('out5',y_dirX5 ,3)
        tf.summary.image('out6',y_dirX6 ,3)
        tf.summary.image('out7',y_dirX7 ,3)
        tf.summary.image('out8',y_dirX8 ,3)
        tf.summary.image('out9',y_dirX9 ,3)
        tf.summary.image('out10',y_dirX10 ,3)
        tf.summary.image('out11',y_dirX11 ,3)
        tf.summary.image('out12',y_dirX12 ,3)
        tf.summary.image('out13',y_dirX13 ,3)

        tf.summary.image('groundtruth1', label_dirX1,3)
        tf.summary.image('groundtruth2', label_dirX2,3)
        tf.summary.image('groundtruth3', label_dirX3,3)
        tf.summary.image('groundtruth4', label_dirX4,3)
        tf.summary.image('groundtruth5', label_dirX5,3)
        tf.summary.image('groundtruth6', label_dirX6,3)
        tf.summary.image('groundtruth7', label_dirX7,3)
        tf.summary.image('groundtruth8', label_dirX8,3)
        tf.summary.image('groundtruth9', label_dirX9,3)
        tf.summary.image('groundtruth10', label_dirX10,3)
        tf.summary.image('groundtruth11', label_dirX11,3)
        tf.summary.image('groundtruth12', label_dirX12,3)
        tf.summary.image('groundtruth13', label_dirX13,3)
        tf.summary.image('groundtruth14', label_dirX14,3)



        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        log_extttt=''
        train_writer = tf.summary.FileWriter(self.LOGDIR + '/train' + log_extttt,graph=tf.get_default_graph())
        validation_writer = tf.summary.FileWriter(self.LOGDIR + '/validation' + log_extttt, graph=sess.graph)
        saver=tf.train.Saver(tf.global_variables(), max_to_keep=1000)

        utils.backup_code(self.LOGDIR)

        '''AdamOptimizer:'''
        with tf.name_scope('averaged_mean_squared_error'):
            [MSE_remove_bg1,mseloss1,mseloss2,mseloss3,mseloss4,mseloss5,
             mseloss6,mseloss7,mseloss8,mseloss9,mseloss10,mseloss11,
             mseloss12,mseloss13,mseloss14] = self.loss_instance.MSE_remove_bg(label1=label1,
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
            cost = tf.reduce_mean(MSE_remove_bg1, name="cost")
        with tf.name_scope('huber'):
            [averaged_huber,hloss1,hloss2,hloss3,hloss4,
             hloss5,hloss6,hloss7,hloss8,hloss9,hloss10,
             hloss11,hloss12,hloss13,hloss14] = self.loss_instance.averaged_huber(label1=label1,
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



            # cost = tf.reduce_mean(averaged_huber, name="averaged_huber")

            # huberloss1 = hloss1
            # huberloss2 = hloss2
            # huberloss3 = hloss3
            # huberloss4 = hloss4
            # huberloss5 = hloss5
            # huberloss6 = hloss6
            # huberloss7 = hloss7
            # huberloss8 = hloss8
            # huberloss9 = hloss9
            # huberloss10 = hloss10
            # huberloss11 = hloss11
            # huberloss12 = hloss12
            # huberloss13 = hloss13
            # huberloss14 = hloss14

        # ========================================================================
        ave_huber = tf.placeholder(tf.float32, name='huber')
        # ave_mse = tf.placeholder(tf.float32, name='mse')
        #
        # huber1_vali = tf.placeholder(tf.float32, name='huber1_vali')
        # huber2_vali = tf.placeholder(tf.float32, name='huber2_vali')
        # huber3_vali = tf.placeholder(tf.float32, name='huber3_vali')
        # huber4_vali = tf.placeholder(tf.float32, name='huber4_vali')
        # huber5_vali = tf.placeholder(tf.float32, name='huber5_vali')
        # huber6_vali = tf.placeholder(tf.float32, name='huber6_vali')
        # huber7_vali = tf.placeholder(tf.float32, name='huber7_vali')
        # huber8_vali = tf.placeholder(tf.float32, name='huber8_vali')
        # huber9_vali = tf.placeholder(tf.float32, name='huber9_vali')
        # huber10_vali = tf.placeholder(tf.float32, name='huber10_vali')
        # huber11_vali = tf.placeholder(tf.float32, name='huber11_vali')
        # huber12_vali = tf.placeholder(tf.float32, name='huber14_vali')
        # huber13_vali = tf.placeholder(tf.float32, name='huber13_vali')
        # huber14_vali = tf.placeholder(tf.float32, name='huber14_vali')
        #
        # mse1_vali = tf.placeholder(tf.float32, name='mse1_vali')
        # mse2_vali = tf.placeholder(tf.float32, name='mse2_vali')
        # mse3_vali = tf.placeholder(tf.float32, name='mse3_vali')
        # mse4_vali = tf.placeholder(tf.float32, name='mse4_vali')
        # mse5_vali = tf.placeholder(tf.float32, name='mse5_vali')
        # mse6_vali = tf.placeholder(tf.float32, name='mse6_vali')
        # mse7_vali = tf.placeholder(tf.float32, name='mse7_vali')
        # mse8_vali = tf.placeholder(tf.float32, name='mse8_vali')
        # mse9_vali = tf.placeholder(tf.float32, name='mse9_vali')
        # mse10_vali = tf.placeholder(tf.float32, name='mse10_vali')
        # mse11_vali = tf.placeholder(tf.float32, name='mse11_vali')
        # mse12_vali = tf.placeholder(tf.float32, name='mse14_vali')
        # mse13_vali = tf.placeholder(tf.float32, name='mse13_vali')
        # mse14_vali = tf.placeholder(tf.float32, name='mse14_vali')
        #
        with tf.name_scope('validation'):
            ave_huber_valii = ave_huber
            # ave_mse_valii = ave_mse

            # huber1_ave = hloss1
            # huber2_ave = hloss2
            # huber3_ave = hloss3
            # huber4_ave = hloss4
            # huber5_ave= hloss5
            # huber6_ave = hloss6
            # huber7_ave = hloss7
            # huber8_ave = hloss8
            # huber9_ave = hloss9
            # huber10_ave = hloss10
            # huber11_ave = hloss11
            # huber12_ave= hloss12
            # huber13_ave = hloss13
            # huber14_ave = hloss14
        #
        #     mse1_vali = mse1_vali
        #     mse2_vali = mse2_vali
        #     mse3_vali = mse3_vali
        #     mse4_vali = mse4_vali
        #     mse5_vali = mse5_vali
        #     mse6_vali = mse6_vali
        #     mse7_vali = mse7_vali
        #     mse8_vali = mse8_vali
        #     mse9_vali = mse9_vali
        #     mse10_vali = mse10_vali
        #     mse11_vali = mse11_vali
        #     mse12_vali = mse12_vali
        #     mse13_vali = mse13_vali
        #     mse14_vali = mse14_vali

        tf.summary.scalar("ave_huber_valii", ave_huber_valii)
        # tf.summary.scalar("ave_mse_valii", ave_mse_valii)

        # tf.summary.scalar("huber1_ave", huber1_ave)
        # tf.summary.scalar("huber2_ave", huber2_ave)
        # tf.summary.scalar("huber3_ave", huber3_ave)
        # tf.summary.scalar("huber4_ave", huber4_ave)
        # tf.summary.scalar("huber5_ave", huber5_ave)
        # tf.summary.scalar("huber6_ave", huber6_ave)
        # tf.summary.scalar("huber7_ave", huber7_ave)
        # tf.summary.scalar("huber8_ave", huber8_ave)
        # tf.summary.scalar("huber9_ave", huber9_ave)
        # tf.summary.scalar("huber10_ave", huber10_ave)
        # tf.summary.scalar("huber11_ave", huber11_ave)
        # tf.summary.scalar("huber12_ave", huber12_ave)
        # tf.summary.scalar("huber13_ave", huber13_ave)
        # tf.summary.scalar("huber14_ave", huber14_ave)
        #
        # tf.summary.scalar("mse1_vali", mse1_vali)
        # tf.summary.scalar("mse2_vali", mse2_vali)
        # tf.summary.scalar("mse3_vali", mse3_vali)
        # tf.summary.scalar("mse4_vali", mse4_vali)
        # tf.summary.scalar("mse5_vali", mse5_vali)
        # tf.summary.scalar("mse6_vali", mse6_vali)
        # tf.summary.scalar("mse7_vali", mse7_vali)
        # tf.summary.scalar("mse8_vali", mse8_vali)
        # tf.summary.scalar("mse9_vali", mse9_vali)
        # tf.summary.scalar("mse10_vali", mse10_vali)
        # tf.summary.scalar("mse11_vali", mse11_vali)
        # tf.summary.scalar("mse12_vali", mse12_vali)
        # tf.summary.scalar("mse13_vali", mse13_vali)
        # tf.summary.scalar("mse14_vali", mse14_vali)


        # ========================================================================

        # with tf.name_scope('psnr'):
        #     [averaged_psnr,psnr1,psnr2,psnr3,psnr4,
        #      psnr5,psnr6,psnr7,psnr8,psnr9,psnr10,
        #      psnr11,psnr12,psnr13,psnr14] = self.loss_instance.averaged_psnr(label1=label1,
        #                                                          label2=label2,
        #                                                          label3=label3,
        #                                                          label4=label4,
        #                                                          label5=label5,
        #                                                          label6=label6,
        #                                                          label7=label7,
        #                                                          label8=label8,
        #                                                          label9=label9,
        #                                                          label10=label10,
        #                                                          label11=label11,
        #                                                          label12=label12,
        #                                                          label13=label13,
        #                                                          label14=label14,
        #                                                          logit1=y[:, :, :, :, 0,np.newaxis],
        #                                                          logit2=y[:, :, :, :, 1,np.newaxis],
        #                                                          logit3=y[:, :, :, :, 2,np.newaxis],
        #                                                          logit4=y[:, :, :, :, 3,np.newaxis],
        #                                                          logit5=y[:, :, :, :, 4,np.newaxis],
        #                                                          logit6=y[:, :, :, :, 5,np.newaxis],
        #                                                          logit7=y[:, :, :, :, 6,np.newaxis],
        #                                                          logit8=y[:, :, :, :, 7,np.newaxis],
        #                                                          logit9=y[:, :, :, :, 8,np.newaxis],
        #                                                          logit10=y[:, :, :, :, 9,np.newaxis],
        #                                                          logit11=y[:, :, :, :, 10,np.newaxis],
        #                                                          logit12=y[:, :, :, :, 11,np.newaxis],
        #                                                          logit13=y[:, :, :, :, 12,np.newaxis],
        #                                                          logit14=y[:, :, :, :, 13,np.newaxis]
        #                                                          )
        # tf.summary.scalar("averaged_huber", averaged_huber)
        # tf.summary.scalar("huberloss1", hloss1)
        # tf.summary.scalar("huberloss2", hloss2)
        # tf.summary.scalar("huberloss3", hloss3)
        # tf.summary.scalar("huberloss4", hloss4)
        # tf.summary.scalar("huberloss5", hloss5)
        # tf.summary.scalar("huberloss6", hloss6)
        # tf.summary.scalar("huberloss7", hloss7)
        # tf.summary.scalar("huberloss8", hloss8)
        # tf.summary.scalar("huberloss9", hloss9)
        # tf.summary.scalar("huberloss10", hloss10)
        # tf.summary.scalar("huberloss11", hloss11)
        # tf.summary.scalar("huberloss12", hloss12)
        # tf.summary.scalar("huberloss13", hloss13)
        # tf.summary.scalar("huberloss14", hloss14)

        # tf.summary.scalar("averaged_mse", averaged_mse)
        # tf.summary.scalar("mseloss1", mseloss1)
        # tf.summary.scalar("mseloss2", mseloss2)
        # tf.summary.scalar("mseloss3", mseloss3)
        # tf.summary.scalar("mseloss4", mseloss4)
        # tf.summary.scalar("mseloss5", mseloss5)
        # tf.summary.scalar("mseloss6", mseloss6)
        # tf.summary.scalar("mseloss7", mseloss7)
        # tf.summary.scalar("mseloss8", mseloss8)
        # tf.summary.scalar("mseloss9", mseloss9)
        # tf.summary.scalar("mseloss10", mseloss10)
        # tf.summary.scalar("mseloss11", mseloss11)
        # tf.summary.scalar("mseloss12", mseloss12)
        # tf.summary.scalar("mseloss13", mseloss13)
        # tf.summary.scalar("mseloss14", mseloss14)

        # tf.summary.scalar("averaged_psnr", averaged_psnr)
        # tf.summary.scalar("PSNR1", psnr1)
        # tf.summary.scalar("PSNR2", psnr2)
        # tf.summary.scalar("PSNR3", psnr3)
        # tf.summary.scalar("PSNR4", psnr4)
        # tf.summary.scalar("PSNR5", psnr5)
        # tf.summary.scalar("PSNR6", psnr6)
        # tf.summary.scalar("PSNR7", psnr7)
        # tf.summary.scalar("PSNR8", psnr8)
        # tf.summary.scalar("PSNR9", psnr9)
        # tf.summary.scalar("PSNR10", psnr10)
        # tf.summary.scalar("PSNR11", psnr11)
        # tf.summary.scalar("PSNR12", psnr12)
        # tf.summary.scalar("PSNR13", psnr13)
        # tf.summary.scalar("PSNR14", psnr14)


        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)



        sess.run(tf.global_variables_initializer())
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
                    acc_validation = 0
                    validation_step = 0
                    dsc_validation=0

                    # ave_huber_loss1=0
                    # ave_huber_loss2=0
                    # ave_huber_loss3=0
                    # ave_huber_loss4=0
                    # ave_huber_loss5=0
                    # ave_huber_loss6=0
                    # ave_huber_loss7=0
                    # ave_huber_loss8=0
                    # ave_huber_loss9=0
                    # ave_huber_loss10=0
                    # ave_huber_loss11=0
                    # ave_huber_loss12=0
                    # ave_huber_loss13=0
                    # ave_huber_loss14=0

                while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):
                        # [vali_mri1,vali_mri2,vali_mri3,vali_mri4,
                        #  vali_mri5, vali_mri6, vali_mri7, vali_mri8,
                        #  vali_perf1,vali_perf2,vali_perf3,
                        #  vali_perf4,vali_perf5,vali_perf6,vali_perf7,
                        #  vali_angio1, vali_angio2, vali_angio3,
                        #  vali_angio4, vali_angio5, vali_angio6,
                        #  vali_angio7]
                        [crush,noncrush,perf,angio]=_image_class_vl.return_patches_vl( validation_step * self.batch_no_validation,
                                                                                                                (validation_step + 1) *self.batch_no_validation,is_tr=False
                                                                                             )
                        if (len(angio)<self.batch_no_validation ) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue
                        # continue


                        [loss_vali,
                         # huber_loss1_vali,huber_loss2_vali,huber_loss3_vali,
                         # huber_loss4_vali,huber_loss5_vali,huber_loss6_vali,
                         # huber_loss7_vali,huber_loss8_vali,huber_loss9_vali,
                         # huber_loss10_vali, huber_loss11_vali, huber_loss12_vali,
                         # huber_loss13_vali, huber_loss14_vali
                         ] = sess.run([cost,
                                                                  # huberloss1,huberloss2,
                                                                  # huberloss3,huberloss4,
                                                                  # huberloss5,huberloss6,
                                                                  # huberloss7,huberloss8,
                                                                  # huberloss9,huberloss10,
                                                                  # huberloss11, huberloss12,
                                                                  # huberloss13,huberloss14
                                                                            ],
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
                                                                    ave_huber: -1,
                                                                    # hloss1: -1,
                                                                    # hloss2: -1,
                                                                    # hloss3: -1,
                                                                    # hloss4: -1,
                                                                    # hloss5: -1,
                                                                    # hloss6: -1,
                                                                    # hloss7: -1,
                                                                    # hloss8: -1,
                                                                    # hloss9: -1,
                                                                    # hloss10: -1,
                                                                    # hloss11: -1,
                                                                    # hloss12: -1,
                                                                    # hloss13: -1,
                                                                    # hloss14: -1,
                                                                    # ave_mse: -1,
                                                                    # mse1_vali: -1,
                                                                    # mse2_vali: -1,
                                                                    # mse3_vali: -1,
                                                                    # mse4_vali: -1,
                                                                    # mse5_vali: -1,
                                                                    # mse6_vali: -1,
                                                                    # mse7_vali: -1,
                                                                    # mse8_vali: -1,
                                                                    # mse9_vali: -1,
                                                                    # mse10_vali: -1,
                                                                    # mse11_vali: -1,
                                                                    # mse12_vali: -1,
                                                                    # mse13_vali: -1,
                                                                    # mse14_vali: -1
                                                                    })
                        loss_validation += loss_vali
                        # ave_huber_loss1 += huber_loss1_vali
                        # ave_huber_loss2 += huber_loss2_vali
                        # ave_huber_loss3 += huber_loss3_vali
                        # ave_huber_loss4 += huber_loss4_vali
                        # ave_huber_loss5 += huber_loss5_vali
                        # ave_huber_loss6 += huber_loss6_vali
                        # ave_huber_loss7 += huber_loss7_vali
                        # ave_huber_loss8 += huber_loss8_vali
                        # ave_huber_loss9 += huber_loss9_vali
                        # ave_huber_loss10 +=huber_loss10_vali
                        # ave_huber_loss11 +=huber_loss11_vali
                        # ave_huber_loss12 +=huber_loss12_vali
                        # ave_huber_loss13 +=huber_loss13_vali
                        # ave_huber_loss14 +=huber_loss14_vali

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
                dsc_validation = dsc_validation / (validation_step)

                # ave_huber_loss1 =ave_huber_loss1 /validation_step
                # ave_huber_loss2=ave_huber_loss2 /validation_step
                # ave_huber_loss3=ave_huber_loss3 /validation_step
                # ave_huber_loss4=ave_huber_loss4 /validation_step
                # ave_huber_loss5=ave_huber_loss5 /validation_step
                # ave_huber_loss6=ave_huber_loss6 /validation_step
                # ave_huber_loss7=ave_huber_loss7 /validation_step
                # ave_huber_loss8=ave_huber_loss8 /validation_step
                # ave_huber_loss9=ave_huber_loss9 /validation_step
                # ave_huber_loss10=ave_huber_loss10/validation_step
                # ave_huber_loss11=ave_huber_loss11/validation_step
                # ave_huber_loss12=ave_huber_loss12/validation_step
                # ave_huber_loss13=ave_huber_loss13/validation_step
                # ave_huber_loss14=ave_huber_loss14/validation_step

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
                                                           ave_huber: loss_validation,
                                                           # hloss1: ave_huber_loss1,
                                                           # hloss2: ave_huber_loss2,
                                                           # hloss3: ave_huber_loss3,
                                                           # hloss4: ave_huber_loss4,
                                                           # hloss5: ave_huber_loss5,
                                                           # hloss6: ave_huber_loss6,
                                                           # hloss7: ave_huber_loss7,
                                                           # hloss8: ave_huber_loss8,
                                                           # hloss9: ave_huber_loss9,
                                                           # hloss10: ave_huber_loss10,
                                                           # hloss11: ave_huber_loss11,
                                                           # hloss12: ave_huber_loss12,
                                                           # hloss13: ave_huber_loss13,
                                                           # hloss14: ave_huber_loss14,
                                                           # ave_mse: -1,
                                                           # mse1_vali: -1,
                                                           # mse2_vali: -1,
                                                           # mse3_vali: -1,
                                                           # mse4_vali: -1,
                                                           # mse5_vali: -1,
                                                           # mse6_vali: -1,
                                                           # mse7_vali: -1,
                                                           # mse8_vali: -1,
                                                           # mse9_vali: -1,
                                                           # mse10_vali: -1,
                                                           # mse11_vali: -1,
                                                           # mse12_vali: -1,
                                                           # mse13_vali: -1,
                                                           # mse14_vali: -1
                                                           })
                validation_writer.add_summary(sum_validation, point)
                print('end of validation---------%d' % (point))


                '''loop for training batches'''
                while(step*self.batch_no<self.no_sample_per_each_itr):

                    # [train_CT_image_patchs, train_GTV_label, train_Penalize_patch,loss_coef_weights] = _image_class.return_patches( self.batch_no)

                    [crush, noncrush, perf, angio] = _image_class.return_patches_tr(self.batch_no)

                    if (len(angio)<self.batch_no):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue



                    [ loss_train1, optimizing,out,
                      # huber_loss1_tr, huber_loss2_tr, huber_loss3_tr,
                      # huber_loss4_tr, huber_loss5_tr, huber_loss6_tr,
                      # huber_loss7_tr, huber_loss8_tr, huber_loss9_tr,
                      # huber_loss10_tr, huber_loss11_tr, huber_loss12_tr,
                      # huber_loss13_tr, huber_loss14_tr
                                                      ] = sess.run([ cost, optimizer,y,
                                                                     # huberloss1, huberloss2,
                                                                     # huberloss3, huberloss4,
                                                                     # huberloss5, huberloss6,
                                                                     # huberloss7, huberloss8,
                                                                     # huberloss9, huberloss10,
                                                                     # huberloss11, huberloss12,
                                                                     # huberloss13, huberloss14
                                                                     ],
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
                                                                    ave_huber: -1,
                                                                    # hloss1: -1,
                                                                    # hloss2: -1,
                                                                    # hloss3: -1,
                                                                    # hloss4: -1,
                                                                    # hloss5: -1,
                                                                    # hloss6: -1,
                                                                    # hloss7: -1,
                                                                    # hloss8: -1,
                                                                    # hloss9: -1,
                                                                    # hloss10: -1,
                                                                    # hloss11: -1,
                                                                    # hloss12: -1,
                                                                    # hloss13: -1,
                                                                    # hloss14: -1,
                                                                    # ave_mse: -1,
                                                                    # huber1_vali: -1,
                                                                    # huber2_vali: -1,
                                                                    # huber3_vali: -1,
                                                                    # huber4_vali: -1,
                                                                    # huber5_vali: -1,
                                                                    # huber6_vali: -1,
                                                                    # huber7_vali: -1,
                                                                    # huber8_vali: -1,
                                                                    # huber9_vali: -1,
                                                                    # huber10_vali: -1,
                                                                    # huber11_vali: -1,
                                                                    # huber12_vali: -1,
                                                                    # huber13_vali: -1,
                                                                    # huber14_vali: -1,
                                                                    # mse1_vali: -1,
                                                                    # mse2_vali: -1,
                                                                    # mse3_vali: -1,
                                                                    # mse4_vali: -1,
                                                                    # mse5_vali: -1,
                                                                    # mse6_vali: -1,
                                                                    # mse7_vali: -1,
                                                                    # mse8_vali: -1,
                                                                    # mse9_vali: -1,
                                                                    # mse10_vali: -1,
                                                                    # mse11_vali: -1,
                                                                    # mse12_vali: -1,
                                                                    # mse13_vali: -1,
                                                                    # mse14_vali: -1
                                                                    })

                    self.x_hist=self.x_hist+1


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
                                                        ave_huber: loss_train1,
                                                        # hloss1: huber_loss1_tr,
                                                        # hloss2: huber_loss2_tr,
                                                        # hloss3: huber_loss3_tr,
                                                        # hloss4: huber_loss4_tr,
                                                        # hloss5: huber_loss5_tr,
                                                        # hloss6: huber_loss6_tr,
                                                        # hloss7: huber_loss7_tr,
                                                        # hloss8: huber_loss8_tr,
                                                        # hloss9: huber_loss9_tr,
                                                        # hloss10:huber_loss10_tr,
                                                        # hloss11:huber_loss11_tr,
                                                        # hloss12:huber_loss12_tr,
                                                        # hloss13:huber_loss13_tr,
                                                        # hloss14:huber_loss14_tr,
                                                        # ave_mse: -1,
                                                        # huber1_vali: -1,
                                                        # huber2_vali: -1,
                                                        # huber3_vali: -1,
                                                        # huber4_vali: -1,
                                                        # huber5_vali: -1,
                                                        # huber6_vali: -1,
                                                        # huber7_vali: -1,
                                                        # huber8_vali: -1,
                                                        # huber9_vali: -1,
                                                        # huber10_vali: -1,
                                                        # huber11_vali: -1,
                                                        # huber12_vali: -1,
                                                        # huber13_vali: -1,
                                                        # huber14_vali: -1,
                                                        # mse1_vali: -1,
                                                        # mse2_vali: -1,
                                                        # mse3_vali: -1,
                                                        # mse4_vali: -1,
                                                        # mse5_vali: -1,
                                                        # mse6_vali: -1,
                                                        # mse7_vali: -1,
                                                        # mse8_vali: -1,
                                                        # mse9_vali: -1,
                                                        # mse10_vali: -1,
                                                        # mse11_vali: -1,
                                                        # mse12_vali: -1,
                                                        # mse13_vali: -1,
                                                        # mse14_vali: -1
                                                        })
                    train_writer.add_summary(sum_train,point)
                    step = step + 1



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



                    itr1 = itr1 + 1
                    point=point+1

            endTime = time.time()
            #==============end of epoch:
            '''saveing model after each epoch'''
            chckpnt_path = os.path.join(self.chckpnt_dir, 'unet.ckpt')
            saver.save(sess, chckpnt_path, global_step=epoch)


            print("End of epoch----> %d, elapsed time: %d" % (epoch, endTime - startTime))


