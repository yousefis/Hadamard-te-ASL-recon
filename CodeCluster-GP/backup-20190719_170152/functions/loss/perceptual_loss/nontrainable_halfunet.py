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

from functions.layers.layers import layers


# !!

class _unet:
    def __init__(self, trainable, reuse, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed = 200
        self.upsampling3d = upsampling()
        self.layers = layers()
        self.trainable = trainable
        self.reuse = reuse

    # ==================================================
    def level_design(self, input, filters1, filters2, is_training, kernel_size, in_size, crop_size, padding1, padding2,
                     scope, reuse):
        with tf.variable_scope(scope):
            conv1 = self.layers.conv3d(input,
                                       filters=filters1,
                                       kernel_size=kernel_size,
                                       padding=padding1,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=scope + '_conv1',
                                       reuse=self.reuse)

            conv2 = self.layers.conv3d(conv1,
                                       filters=filters2,
                                       kernel_size=kernel_size,
                                       padding=padding2,
                                       dilation_rate=2,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=scope + '_conv2',
                                       reuse=self.reuse)

            crop = conv2[:,
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            return conv2, crop

    # ==================================================
    def noisy_input(self, img_row1, is_training):
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
            rnd = s.run(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed),
                                   5))  # , seed=int(time.time())))

        flip_lr_img_row1 = tf.cond(tf.logical_and(is_training, rnd),
                                   lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row1, 4)), axis=4)
                                   , lambda: img_row1)

        return flip_lr_img_row1

    # ==================================================
    def unet(self, img_row1, input_dim, is_training):
        # in_size0 = tf.to_int32(0)
        # in_size1 = tf.to_int32(input_dim)
        # in_size2 = tf.to_int32(in_size1 - 3 * 2)  # conv stack
        # in_size3 = tf.to_int32((in_size2 - 2 * 2))  # level_design1
        # in_size4 = tf.to_int32(in_size3 / 2 - 2 * 2)  # downsampleing1+level_design2
        # in_size5 = tf.to_int32(in_size4 / 2 - 2 * 2)  # downsampleing2+level_design3
        # crop_size0 = tf.to_int32(0)
        # crop_size1 = tf.to_int32(2 * in_size5 + 1)
        # crop_size2 = tf.to_int32(2 * (crop_size1 - 2 * 2) + 1)

        in_size0 = tf.to_int32(0)
        in_size1 = tf.to_int32(input_dim)
        in_size2 = tf.to_int32(in_size1)  # conv stack
        in_size3 = tf.to_int32((in_size2))  # level_design1
        in_size4 = tf.to_int32(in_size3 / 2)  # downsampleing1+level_design2
        in_size5 = tf.to_int32(in_size4 / 2 - 4)  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0)
        crop_size1 = tf.to_int32(2 * in_size5 + 1)
        crop_size2 = tf.to_int32(2 * (crop_size1 - 4) + 1)

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
        # == == == == == == == == == == == == == == == == == == == == == ==
        # level 1 of unet
        with tf.variable_scope('unet_LD_DS1'):
            conv1 = self.layers.conv3d(img_row1,
                                       filters=16,
                                       kernel_size=3,
                                       padding='same',
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope='unet_LD_DS1' + '_conv1',
                                       reuse=self.reuse)

            conv2 = self.layers.conv3d(conv1,
                                       filters=32,
                                       kernel_size=3,
                                       padding='same',
                                       dilation_rate=2,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope='unet_LD_DS1' + '_conv2',
                                       reuse=self.reuse)

            crop = conv2[:,
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]


        self.pool1 = tf.layers.max_pooling3d(inputs=self.level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [self.level_ds2, crop2] = self.level_design(self.pool1, filters1=32, filters2=64,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               scope='unet_LD_DS2',
                                               reuse=self.reuse)
        self.pool2 = tf.layers.max_pooling3d(inputs=self.level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [self.level_us1, crop0] = self.level_design(self.pool2, filters1=64, filters2=128,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='valid',
                                               scope='unet_LD_US1',
                                               reuse=self.reuse)






