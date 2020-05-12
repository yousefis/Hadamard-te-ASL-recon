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
from functions.network.m_densenet_02 import synth_seg_densenet

# calculate the dice coefficient
from functions.threads.extractor_thread import _extractor_thread
from functions.threads.fill_thread import fill_thread
from functions.threads.read_thread import read_thread


# --------------------------------------------------------------------------------------------------------
class call_synth_seg_densenet:
    def __init__(self,data,  sample_no,validation_samples,no_sample_per_each_itr,
                 train_tag, validation_tag, test_tag,img_name,label_name,torso_tag,log_tag,min_range,max_range,
                 Logs,fold,Server='DL',newdataset=False):
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

        self.label_patchs_size =63
        self.patch_window = 77#89
        self.sample_no = sample_no
        self.batch_no =9
        self.batch_no_validation = 9
        self.validation_samples = validation_samples
        self.display_step = 100
        self.display_validation_step = 1
        self.total_epochs = 10
        self.loss_instance=_loss_func()

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
        self.x_hist=0

        self.fold=fold


        logger.set_log_file(self.parent_path + self.Logs + self.log_ext + '/log_file' + str(fold))

    def save_file(self,file_name,txt):
        with open(file_name, 'a') as file:
            file.write(txt)
    def run_net(self):
        _rd = _read_data(data=self.data,train_tag=self.train_tag, validation_tag=self.validation_tag, test_tag=self.test_tag,
                         img_name=self.img_name, label_name=self.label_name,torso_tag=self.torso_tag,dataset_path=self.data_path)

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

        # seg_label = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])




        is_training = tf.placeholder(tf.bool, name='is_training')
        input_dim = tf.placeholder(tf.int32, name='input_dim')

        synth_densenet=synth_seg_densenet()

        y, mask_angio, seg_angio1,mask_perf ,seg_perf1=synth_densenet.densenet( img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4, img_row5=img_row5,
                     img_row6=img_row6, img_row7=img_row7, img_row8=img_row8,input_dim=input_dim,is_training=is_training)




        tf.summary.image('out0',((y[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])), 3)
        tf.summary.image('out1',((y[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis])), 3)
        tf.summary.image('out2',((y[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis])), 3)
        tf.summary.image('out3',((y[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis])), 3)
        tf.summary.image('out4',((y[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis])), 3)
        tf.summary.image('out5',((y[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis])), 3)
        tf.summary.image('out6',((y[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis])), 3)

        tf.summary.image('out7' ,((y[:, int(self.label_patchs_size / 2), :, :, 7, np.newaxis])), 3)
        tf.summary.image('out8' ,((y[:, int(self.label_patchs_size / 2), :, :, 8, np.newaxis])), 3)
        tf.summary.image('out9' ,((y[:, int(self.label_patchs_size / 2), :, :, 9, np.newaxis])), 3)
        tf.summary.image('out10',((y[:, int(self.label_patchs_size / 2), :, :, 10, np.newaxis])), 3)
        tf.summary.image('out11',((y[:, int(self.label_patchs_size / 2), :, :, 11, np.newaxis])), 3)
        tf.summary.image('out12',((y[:, int(self.label_patchs_size / 2), :, :, 12, np.newaxis])), 3)
        tf.summary.image('out13',((y[:, int(self.label_patchs_size / 2), :, :, 13, np.newaxis])), 3)

        tf.summary.image('out14', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])), 3)
        tf.summary.image('out15', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis])), 3)
        tf.summary.image('out16', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis])), 3)
        tf.summary.image('out17', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis])), 3)
        tf.summary.image('out18', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis])), 3)
        tf.summary.image('out19', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis])), 3)
        tf.summary.image('out20', ((seg_angio1[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis])), 3)

        tf.summary.image('out21', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])), 3)
        tf.summary.image('out22', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis])), 3)
        tf.summary.image('out23', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis])), 3)
        tf.summary.image('out24', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis])), 3)
        tf.summary.image('out25', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis])), 3)
        tf.summary.image('out26', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis])), 3)
        tf.summary.image('out27', ((seg_perf1[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis])), 3)

        tf.summary.image('out28', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])), 3)
        tf.summary.image('out29', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis])), 3)
        tf.summary.image('out30', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis])), 3)
        tf.summary.image('out31', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis])), 3)
        tf.summary.image('out32', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis])), 3)
        tf.summary.image('out33', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis])), 3)
        tf.summary.image('out34', ((mask_angio[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis])), 3)

        tf.summary.image('out35', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis])), 3)
        tf.summary.image('out36', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 1, np.newaxis])), 3)
        tf.summary.image('out37', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 2, np.newaxis])), 3)
        tf.summary.image('out38', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 3, np.newaxis])), 3)
        tf.summary.image('out39', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 4, np.newaxis])), 3)
        tf.summary.image('out40', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 5, np.newaxis])), 3)
        tf.summary.image('out41', ((mask_perf[:, int(self.label_patchs_size / 2), :, :, 6, np.newaxis])), 3)

        tf.summary.image('groundtruth1', (label1[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth2', (label2[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth3', (label3[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth4', (label4[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth5', (label5[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth6', (label6[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth7', (label7[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)

        tf.summary.image('groundtruth8', (label8[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth9', (label9[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth10',(label10[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth11',(label11[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth12',(label12[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth13',(label13[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)
        tf.summary.image('groundtruth14',(label14[:, int(self.label_patchs_size / 2), :, :, 0, np.newaxis]), 3)


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
            [averaged_huber,mseloss1,mseloss2,mseloss3,mseloss4,mseloss5,
             mseloss6,mseloss7,mseloss8,mseloss9,mseloss10,mseloss11,
             mseloss12,mseloss13,mseloss14] = self.loss_instance.synth_seg_huber(label1=label1,
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
                                                                 seg_angio_logit=mask_angio,
                                                                 seg_angio_label=seg_angio1,
                                                                 seg_perf_logit=mask_perf,
                                                                 seg_perf_label=seg_perf1,
                                                                 logit1=y[:, :, :, :, 0,np.newaxis],
                                                                 logit2=y[:, :, :, :, 1,np.newaxis],
                                                                 logit3=y[:, :, :, :, 2,np.newaxis],
                                                                 logit4=y[:, :, :, :, 3,np.newaxis],
                                                                 logit5=y[:, :, :, :, 4,np.newaxis],
                                                                 logit6=y[:, :, :, :, 5,np.newaxis],
                                                                 logit7=y[:, :, :, :, 6,np.newaxis],
                                                                 logit8= y[:, :, :, :, 0,np.newaxis],
                                                                 logit9= y[:, :, :, :, 1,np.newaxis],
                                                                 logit10=y[:, :, :, :, 2,np.newaxis],
                                                                 logit11=y[:, :, :, :, 3,np.newaxis],
                                                                 logit12=y[:, :, :, :, 4,np.newaxis],
                                                                 logit13=y[:, :, :, :, 5,np.newaxis],
                                                                 logit14=y[:, :, :, :, 6,np.newaxis]
                                                                 )
            cost = tf.reduce_mean(averaged_huber, name="cost")

        ave_huber = tf.placeholder(tf.float32, name='huber')
        with tf.name_scope('validation'):
            ave_huber_valii = ave_huber

        tf.summary.scalar("ave_huber_valii", ave_huber_valii)
        # ========================================================================



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
                        if (len(crush)<self.batch_no_validation or
                            len(noncrush)<self.batch_no_validation or
                            len(perf)<self.batch_no_validation or
                            len(angio) < self.batch_no_validation ) :
                            _read_thread_vl.resume()
                            time.sleep(0.5)
                            continue


                        [loss_vali,] = sess.run([cost, ],
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
                                                           ave_huber: loss_validation,

                                                           })
                validation_writer.add_summary(sum_validation, point)
                print('end of validation---------%d' % (point))


                '''loop for training batches'''
                while(step*self.batch_no<self.no_sample_per_each_itr):

                    [crush, noncrush, perf, angio] = _image_class.return_patches_tr(self.batch_no)

                    if (len(crush)<self.batch_no or
                            len(noncrush)<self.batch_no or
                            len(perf)<self.batch_no or
                            len(angio) < self.batch_no ):
                        time.sleep(0.5)
                        _read_thread.resume()
                        continue



                    [ loss_train1, optimizing,out,
                                                      ] = sess.run([ cost, optimizer,y,
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


