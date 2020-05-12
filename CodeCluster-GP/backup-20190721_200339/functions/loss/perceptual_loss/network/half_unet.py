import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import time
from functions.layers.upsampling import upsampling


# !!

class _half_unet:
    def __init__(self,trainable, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed=200
        self.upsampling3d=upsampling()

        self.trainable=trainable
    # ========================
    def conv3d(self,input,filters,kernel_size,padding,dilation_rate,is_training,trainable,scope,reuse):
        conv = tf.layers.conv3d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 activation=None,
                                 dilation_rate=dilation_rate,
                                 trainable=trainable,
                                 name=scope+'_conv3d',
                                reuse=reuse
                                 )
        bn=tf.layers.batch_normalization(conv,
                                       training=is_training,
                                       renorm=False,
                                       trainable=trainable,
                                       name=scope + '_bn',
                                         reuse=reuse)
        bn = tf.nn.relu(bn)
        return bn
    # ========================
    def convolution_stack(self, input, stack_name, filters, kernel_size, padding, is_training,reuse):
        with tf.name_scope(stack_name):
            conv1 = tf.cond(self.trainable,
                            lambda :self.conv3d(input,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=1,
                                                is_training=is_training,
                                                trainable=True,
                                                scope='stack1_true',
                                                reuse=reuse),
                            lambda: self.conv3d(input,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=1,
                                                is_training=is_training,
                                                trainable=False,
                                                scope='stack1_false',
                                                reuse=reuse)
                            )
            conv2 = tf.cond(self.trainable,
                            lambda: self.conv3d(conv1,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=2,
                                                is_training=is_training,
                                                trainable=True,
                                                scope='stack2_true',
                                                reuse=reuse),
                            lambda: self.conv3d(conv1, filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                dilation_rate=2,
                                                is_training=is_training,
                                                trainable=False,
                                                scope='stack2_false',
                                                reuse=reuse)
                            )

            conv2=tf.concat([input,conv2],4)
            return conv2
    #==================================================
    def level_design(self,input,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2,scope,reuse):
        with tf.name_scope(scope):
            conv1 =tf.cond(self.trainable,
                            lambda: self.conv3d(input,
                                                filters=filters1,
                                                kernel_size=kernel_size,
                                                padding=padding1,
                                                dilation_rate=1,
                                                is_training=is_training,
                                                trainable=True,
                                                scope=scope+'_true',
                                                reuse=reuse),
                            lambda: self.conv3d(input,
                                                filters=filters1,
                                                kernel_size=kernel_size,
                                                padding=padding1,
                                                dilation_rate=1,
                                                is_training=is_training,
                                                trainable=False,
                                                scope=scope+'_false',
                                                reuse=reuse)
                            )


            conv2 =tf.cond(self.trainable,
                            lambda: self.conv3d(conv1,
                                                filters=filters2,
                                                kernel_size=kernel_size,
                                                padding=padding2,
                                                dilation_rate=2,
                                                is_training=is_training,
                                                trainable=True,
                                                scope=scope+'_true_2',
                                                reuse=reuse),
                            lambda: self.conv3d(conv1, filters=filters2,
                                                kernel_size=kernel_size,
                                                padding=padding2,
                                                dilation_rate=2,
                                                is_training=is_training,
                                                trainable=False,
                                                scope=scope+'_false_2',
                                                reuse=reuse)
                            )

            crop = conv2[:,
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]


            return conv2,crop

    #==================================================
    def noisy_input(self,img_row1,is_training):
        with tf.Session() as s:
            rnd = s.run(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed))  # , seed=int(time.time())))

        noisy_img_row1 = tf.cond(is_training,
                                 lambda: img_row1 + tf.round(tf.random_normal(tf.shape(img_row1), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row1)

        return noisy_img_row1

        # ==================================================
    def flip_lr_input(self, img_row1,
                    is_training):
        with tf.Session() as s:
            rnd = s.run(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed),5))  # , seed=int(time.time())))

        flip_lr_img_row1 = tf.cond( tf.logical_and(is_training,rnd),
                                 lambda:  tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row1,4)),axis=4)
                                 , lambda: img_row1)

        return flip_lr_img_row1
    #==================================================
    def half_unet(self, img_row1, input_dim, is_training,reuse):
        # in_size0 = tf.to_int32(0)
        # in_size1 = tf.to_int32(input_dim)
        # in_size2 = tf.to_int32(in_size1 - 3 * 2)  # conv stack
        # in_size3 = tf.to_int32((in_size2 - 2 * 2))  # level_design1
        # in_size4 = tf.to_int32(in_size3 / 2 - 2 * 2)  # downsampleing1+level_design2
        # in_size5 = tf.to_int32(in_size4 / 2 - 2 * 2)  # downsampleing2+level_design3
        # crop_size0 = tf.to_int32(0)
        # crop_size1 = tf.to_int32(2 * in_size5 + 1)
        # crop_size2 = tf.to_int32(2 * (crop_size1 - 2 * 2) + 1)

        # in_size0 = tf.to_int32(0)
        # in_size1 = tf.to_int32(input_dim)
        # in_size2 = tf.to_int32(in_size1)  # conv stack
        # in_size3 = tf.to_int32((in_size2))  # level_design1
        # in_size4 = tf.to_int32(in_size3 / 2)  # downsampleing1+level_design2
        # in_size5 = tf.to_int32(in_size4 / 2 - 4)  # downsampleing2+level_design3
        # crop_size0 = tf.to_int32(0)
        # crop_size1 = tf.to_int32(2 * in_size5 + 1)
        # crop_size2 = tf.to_int32(2 * (crop_size1 - 4) + 1)

        # input_dim=55
        # in_size0 = (0)
        # in_size1 = (input_dim)
        # in_size2 = (in_size1 )  # conv stack
        # in_size3 = ((in_size2))  # level_design1
        # in_size4 = int(in_size3 / 2 )  # downsampleing1+level_design2
        # in_size5 = int(in_size4 / 2 -4)  # downsampleing2+level_design3
        # crop_size0 = (0)
        # crop_size1 = (2 * in_size5 + 1)
        # crop_size2 = (2 * (crop_size1-4 ) + 1)
        # final_layer = crop_size2-2*2
        in_size0 = tf.to_int32(0)
        in_size1 = (input_dim)
        in_size2 = (in_size1 )  # conv stack
        in_size3 = ((in_size2))  # level_design1
        in_size4 = tf.to_int32(in_size3 / 2 )  # downsampleing1+level_design2
        in_size5 = tf.to_int32(in_size4 / 2 -4)  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0)
        crop_size1 = tf.to_int32(2 * in_size5 + 1)
        crop_size2 = tf.to_int32(2 * (crop_size1-4 ) + 1)

        img_row1=self.noisy_input( img_row1,is_training)

        img_row1=self.flip_lr_input(img_row1,is_training)
        filters=8
        stack1=self.convolution_stack(input=img_row1,
                                      stack_name='stack_1',
                                      filters=filters,
                                      kernel_size=3,
                                      padding='same',
                                      is_training=is_training,
                                      reuse=reuse)



        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack1,  filters1=16, filters2=32,
                                               is_training=is_training, kernel_size=3, in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               scope='DS1',
                                               reuse=reuse)
        pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1,  filters1=32, filters2=64,
                                               is_training=is_training, kernel_size=3, in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               scope='DS2',
                                               reuse=reuse)
        pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [level_us1, crop0] = self.level_design(pool2, filters1=64, filters2=128,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='valid',
                                               scope='US1',
                                               reuse=reuse)




        return  level_us1#,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6

