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
from functions.network.forked_densenet_2 import _forked_densenet2
# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread
from functions.layers.downsampler import downsampler


# --------------------------------------------------------------------------------------------------------
class forked_synthesizing_net22:
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
        # ==================================
        settings.validation_totalimg_patch=validation_samples
        self.downsampler=downsampler()
        # ==================================
        self.learning_decay = .95
        self.learning_rate = 1E-5
        self.beta_rate = 0.05

        self.img_padded_size = 519
        self.seg_size = 505
        self.min_range = min_range
        self.max_range = max_range

        self.label_patchs_size =47
        self.patch_window = 79#89
        self.sample_no = sample_no
        self.batch_no =9
        self.batch_no_validation = 9
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance=_loss_func()
        Server='Shark'
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
    def subject_downsampler(self,label1,label2,label3,label4,label5,label6,label7
                    , label8, label9, label10, label11, label12, label13, label14):
        # perfution 7time points 1/2 scale
        label1_scale2 = self.downsampler.downsampler_gpu(input=label1, down_scale=2)
        label2_scale2 = self.downsampler.downsampler_gpu(input=label2, down_scale=2)
        label3_scale2 = self.downsampler.downsampler_gpu(input=label3, down_scale=2)
        label4_scale2 = self.downsampler.downsampler_gpu(input=label4, down_scale=2)
        label5_scale2 = self.downsampler.downsampler_gpu(input=label5, down_scale=2)
        label6_scale2 = self.downsampler.downsampler_gpu(input=label6, down_scale=2)
        label7_scale2 = self.downsampler.downsampler_gpu(input=label7, down_scale=2)

        # angio 7time points 1/2 scale
        label8_scale2 = self.downsampler.downsampler_gpu(input=label8, down_scale=2)
        label9_scale2 = self.downsampler.downsampler_gpu(input=label9, down_scale=2)
        label10_scale2 = self.downsampler.downsampler_gpu(input=label10, down_scale=2)
        label11_scale2 = self.downsampler.downsampler_gpu(input=label11, down_scale=2)
        label12_scale2 = self.downsampler.downsampler_gpu(input=label12, down_scale=2)
        label13_scale2 = self.downsampler.downsampler_gpu(input=label13, down_scale=2)
        label14_scale2 = self.downsampler.downsampler_gpu(input=label14, down_scale=2)

        # perfution 7time points 1/4 scale
        label1_scale4 = self.downsampler.downsampler_gpu(input=label1, down_scale=4)
        label2_scale4 = self.downsampler.downsampler_gpu(input=label2, down_scale=4)
        label3_scale4 = self.downsampler.downsampler_gpu(input=label3, down_scale=4)
        label4_scale4 = self.downsampler.downsampler_gpu(input=label4, down_scale=4)
        label5_scale4 = self.downsampler.downsampler_gpu(input=label5, down_scale=4)
        label6_scale4 = self.downsampler.downsampler_gpu(input=label6, down_scale=4)
        label7_scale4 = self.downsampler.downsampler_gpu(input=label7, down_scale=4)

        # angio 7time points 1/4 scale
        label8_scale4 = self.downsampler.downsampler_gpu(input=label8, down_scale=4)
        label9_scale4 = self.downsampler.downsampler_gpu(input=label9, down_scale=4)
        label10_scale4 = self.downsampler.downsampler_gpu(input=label10, down_scale=4)
        label11_scale4 = self.downsampler.downsampler_gpu(input=label11, down_scale=4)
        label12_scale4 = self.downsampler.downsampler_gpu(input=label12, down_scale=4)
        label13_scale4 = self.downsampler.downsampler_gpu(input=label13, down_scale=4)
        label14_scale4 = self.downsampler.downsampler_gpu(input=label14, down_scale=4)
        return label1_scale2,label2_scale2,label3_scale2,label4_scale2,label5_scale2,label6_scale2,label7_scale2, \
               label1_scale4, label2_scale4, label3_scale4, label4_scale4, label5_scale4, label6_scale4, label7_scale4, \
               label8_scale2, label9_scale2, label10_scale2, label11_scale2, label12_scale2, label13_scale2, label14_scale2, \
               label8_scale4, label9_scale4, label10_scale4, label11_scale4, label12_scale4, label13_scale4, label14_scale4

    def run_net(self):
        _rd = _read_data(data=self.data,train_tag=self.train_tag, validation_tag=self.validation_tag, test_tag=self.test_tag,
                         img_name=self.img_name, label_name=self.label_name,torso_tag=self.torso_tag,dataset_path=self.data_path)

        self.alpha_coeff=1
        '''read path of the images for train, test, and validation'''
        train_data, validation_data, test_data=_rd.read_data_path()

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
        #
        # img_row1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row1')
        # img_row2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row2')
        # img_row3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row3')
        # img_row4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row4')
        # img_row5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row5')
        # img_row6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row6')
        # img_row7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row7')
        # img_row8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.patch_window,self.patch_window,self.patch_window, 1],name='img_row8')
        #
        # #perfution 7time points original scale
        # label1 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label1')
        # label2 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label2')
        # label3 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label3')
        # label4 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label4')
        # label5 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label5')
        # label6 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label6')
        # # angio 7time points original scale
        # label7 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label7')
        # label8 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label8')
        # label9 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label9')
        # label10 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label10')
        # label11 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label11')
        # label12 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label12')
        # label13 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label13')
        # label14 = tf.placeholder(tf.float32, shape=[self.batch_no,self.label_patchs_size,self.label_patchs_size,self.label_patchs_size, 1],name='label14')


        img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row1')
        img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row2')
        img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row3')
        img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row4')
        img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row5')
        img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row6')
        img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row7')
        img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row8')

        label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label1')
        label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label2')
        label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label3')
        label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label4')
        label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label5')
        label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label6')
        label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label7')
        label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label8')
        label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label9')
        label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label10')
        label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label11')
        label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label12')
        label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label13')
        label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label14')




        is_training = tf.placeholder(tf.bool, name='is_training')
        input_dim = tf.placeholder(tf.int32, name='input_dim')

        forked_densenet=_forked_densenet2()

        perf_y,perf_loss_fm1,perf_loss_fm2,\
        angio_y,angio_loss_fm1,angio_loss_fm2=forked_densenet.densenet( img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4, img_row5=img_row5,
                     img_row6=img_row6, img_row7=img_row7, img_row8=img_row8,input_dim=input_dim,is_training=is_training)

        y_dirX =  ((perf_y[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]))
        y_dirX1 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis]))
        y_dirX2 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis]))
        y_dirX3 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis]))
        y_dirX4 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis]))
        y_dirX5 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis]))
        y_dirX6 = ((perf_y[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis]))
        y_dirX7 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]))
        y_dirX8 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis]))
        y_dirX9 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis]))
        y_dirX10 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis]))
        y_dirX11 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis]))
        y_dirX12 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis]))
        y_dirX13 = ((angio_y[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis]))


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
        with tf.name_scope('MSE_perf_angio'):
            MSE_perf_angio= self.loss_instance.MSE_perf_forked_angio(label1=label1,
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
                                                                 logit1=perf_y[:, :, :, :, 0,np.newaxis],
                                                                 logit2=perf_y[:, :, :, :, 1,np.newaxis],
                                                                 logit3=perf_y[:, :, :, :, 2,np.newaxis],
                                                                 logit4=perf_y[:, :, :, :, 3,np.newaxis],
                                                                 logit5=perf_y[:, :, :, :, 4,np.newaxis],
                                                                 logit6=perf_y[:, :, :, :, 5,np.newaxis],
                                                                 logit7=perf_y[:, :, :, :, 6,np.newaxis],
                                                                 logit8=angio_y[:, :, :, :, 0,np.newaxis],
                                                                 logit9=angio_y[:, :, :, :, 1,np.newaxis],
                                                                 logit10=angio_y[:, :, :, :, 2,np.newaxis],
                                                                 logit11=angio_y[:, :, :, :, 3,np.newaxis],
                                                                 logit12=angio_y[:, :, :, :, 4,np.newaxis],
                                                                 logit13=angio_y[:, :, :, :, 5,np.newaxis],
                                                                 logit14=angio_y[:, :, :, :, 6,np.newaxis],
                                                                 perf_loss_fm1=perf_loss_fm1,
                                                                 perf_loss_fm2=perf_loss_fm2,
                                                                 angio_loss_fm1=angio_loss_fm1,
                                                                 angio_loss_fm2=angio_loss_fm2,
                                                                  )
            cost = tf.reduce_mean(MSE_perf_angio, name="cost")
        tf.summary.scalar("cost", MSE_perf_angio)

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


                while (validation_step * self.batch_no_validation < settings.validation_totalimg_patch):

                        [crush,noncrush,perf,angio]=_image_class_vl.return_patches_vl( validation_step * self.batch_no_validation,
                                                                                                                (validation_step + 1) *self.batch_no_validation,is_tr=False
                                                                                             )
                        if (len(angio)<self.batch_no_validation ) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue
                        # continue


                        [loss_vali ] = sess.run([cost],
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


                                                                    })
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


                    [ loss_train1, optimizing,out,] = sess.run([ cost, optimizer,perf_y,

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


