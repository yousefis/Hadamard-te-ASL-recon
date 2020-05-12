import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import time


# !!

class _unet:
    def __init__(self, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed=200

    # ========================
    def convolution_stack(self, input, stack_name, filters, kernel_size, padding, is_training):
        with tf.name_scope(stack_name):
            conv1 = tf.layers.conv3d(input,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        activation=None,
                                        dilation_rate=1,
                                        )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)
            conv1 = bn

            conv2 = tf.layers.conv3d(conv1,
                                     filters=filters,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     activation=None,
                                     dilation_rate=1)
            bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)
            conv2 = bn
            conv2=tf.concat([input,conv2],4)
            # conv3 = tf.layers.conv3d(conv2,
            #                          filters=filters,
            #                          kernel_size=kernel_size,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            # bn = tf.nn.relu(bn)
            # conv3 = bn
            return conv2
    #==================================================
    def level_design(self,input,level_name,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2):
        with tf.name_scope(level_name):
            conv1 = tf.layers.conv3d(input,
                                     filters=filters1,
                                     kernel_size=kernel_size,
                                     padding=padding1,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)
            conv1 = bn

            conv2 = tf.layers.conv3d(conv1,
                                     filters=filters2,
                                     kernel_size=kernel_size,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=2,
                                     )
            bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)
            conv2 = bn

            crop = conv2[:,
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]


            return conv2,crop

    #==================================================
    def noisy_input(self,img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,is_training):
        with tf.Session() as s:
            rnd = s.run(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed))  # , seed=int(time.time())))

        noisy_img_row1 = tf.cond(is_training,
                                 lambda: img_row1 + tf.round(tf.random_normal(tf.shape(img_row1), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row1)
        noisy_img_row2 = tf.cond(is_training,
                                 lambda: img_row2 + tf.round(tf.random_normal(tf.shape(img_row2), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row2)
        noisy_img_row3 = tf.cond(is_training,
                                 lambda: img_row3 + tf.round(tf.random_normal(tf.shape(img_row3), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row3)
        noisy_img_row4 = tf.cond(is_training,
                                 lambda: img_row4 + tf.round(tf.random_normal(tf.shape(img_row4), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row4)
        noisy_img_row5 = tf.cond(is_training,
                                 lambda: img_row5 + tf.round(tf.random_normal(tf.shape(img_row5), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row5)
        noisy_img_row6 = tf.cond(is_training,
                                 lambda: img_row6 + tf.round(tf.random_normal(tf.shape(img_row6), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row6)
        noisy_img_row7 = tf.cond(is_training,
                                 lambda: img_row7 + tf.round(tf.random_normal(tf.shape(img_row7), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row7)
        noisy_img_row8 = tf.cond(is_training,
                                 lambda: img_row8 + tf.round(tf.random_normal(tf.shape(img_row8), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row8)
        return noisy_img_row1,noisy_img_row2,noisy_img_row3,noisy_img_row4,\
               noisy_img_row5,noisy_img_row6,noisy_img_row7,noisy_img_row8

        # ==================================================
    def flip_lr_input(self, img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,
                    is_training):
        with tf.Session() as s:
            rnd = s.run(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed),5))  # , seed=int(time.time())))

        flip_lr_img_row1 = tf.cond( tf.logical_and(is_training,rnd),
                                 lambda:  tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row1,4)),axis=4)
                                 , lambda: img_row1)
        flip_lr_img_row2 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row2,4)),axis=4)
                                 , lambda: img_row2)
        flip_lr_img_row3 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row3,4)),axis=4)
                                 , lambda: img_row3)
        flip_lr_img_row4 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row4,4)),axis=4)
                                 , lambda: img_row4)
        flip_lr_img_row5 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row5,4)),axis=4)
                                 , lambda: img_row5)
        flip_lr_img_row6 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row6,4)),axis=4)
                                 , lambda: img_row6)
        flip_lr_img_row7 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row7,4)),axis=4)
                                 , lambda: img_row7)
        flip_lr_img_row8 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row8,4)),axis=4)
                                 , lambda: img_row8)
        return flip_lr_img_row1, flip_lr_img_row2, flip_lr_img_row3, flip_lr_img_row4, \
               flip_lr_img_row5, flip_lr_img_row6, flip_lr_img_row7, flip_lr_img_row8
    #==================================================
    def unet(self, img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8, input_dim, is_training):
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
        in_size5 = tf.to_int32(in_size4 / 2 - 6)  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0)
        crop_size1 = tf.to_int32(2 * in_size5 + 1)
        crop_size2 = tf.to_int32(2 * (crop_size1 - 6) + 1)

        # input_dim=55
        # in_size0 = (0)
        # in_size1 = (input_dim)
        # in_size2 = (in_size1 )  # conv stack
        # in_size3 = ((in_size2))  # level_design1
        # in_size4 = int(in_size3 / 2 )  # downsampleing1+level_design2
        # in_size5 = int(in_size4 / 2 -6)  # downsampleing2+level_design3
        # crop_size0 = (0)
        # crop_size1 = (2 * in_size5 + 1)
        # crop_size2 = (2 * (crop_size1 -6) + 1)
        # final_layer = crop_size2-2*2

        [img_row1, img_row2, img_row3, img_row4,
         img_row5, img_row6, img_row7, img_row8]=self.noisy_input( img_row1, img_row2, img_row3, img_row4,
                                                                   img_row5, img_row6, img_row7, img_row8,
                                                                   is_training)

        [img_row1, img_row2, img_row3, img_row4,
         img_row5, img_row6, img_row7, img_row8]=self.flip_lr_input(img_row1, img_row2, img_row3, img_row4,
                                                                    img_row5, img_row6, img_row7, img_row8,
                                                                    is_training)
        filters=8
        stack1=self.convolution_stack(input=img_row1, stack_name='stack_1', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack2=self.convolution_stack(input=img_row2, stack_name='stack_2', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack3=self.convolution_stack(input=img_row3, stack_name='stack_3', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack4=self.convolution_stack(input=img_row4, stack_name='stack_4', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack5=self.convolution_stack(input=img_row5, stack_name='stack_5', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack6=self.convolution_stack(input=img_row6, stack_name='stack_6', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack7=self.convolution_stack(input=img_row7, stack_name='stack_7', filters=filters, kernel_size=3, padding='same', is_training=is_training)
        stack8=self.convolution_stack(input=img_row8, stack_name='stack_8', filters=filters, kernel_size=3, padding='same', is_training=is_training)

        stack_concat = tf.concat([stack1, stack2], 4)
        stack_concat = tf.concat([stack_concat, stack3], 4)
        stack_concat = tf.concat([stack_concat, stack4], 4)
        stack_concat = tf.concat([stack_concat, stack5], 4)
        stack_concat = tf.concat([stack_concat, stack6], 4)
        stack_concat = tf.concat([stack_concat, stack7], 4)
        stack_concat = tf.concat([stack_concat, stack8], 4)
        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack_concat, 'level_ds1', filters1=8, filters2=16,
                                               is_training=is_training, kernel_size=3, in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',padding2='same')
        pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1, 'level_ds2', filters1=25, filters2=32,
                                               is_training=is_training, kernel_size=3, in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same', padding2='same')
        pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [level_us1, crop0] = self.level_design(pool2, 'level_us1', filters1=32, filters2=40,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='valid', padding2='valid')

        deconv1 = tf.layers.conv3d_transpose(level_us1,
                                             filters=40,
                                             kernel_size=3,
                                             strides=(2, 2, 2),
                                             padding='valid',
                                             use_bias=False)
        conc12 = tf.concat([crop2, deconv1], 4)

        # level 2 of unet
        [level_us2, crop0] = self.level_design(conc12, 'level_us2', filters1=45, filters2=32,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='valid', padding2='valid')
        deconv2 = tf.layers.conv3d_transpose(level_us2,
                                             filters=32,
                                             kernel_size=3,
                                             strides=(2, 2, 2),
                                             padding='valid',
                                             use_bias=False)
        conc23 = tf.concat([crop1, deconv2], 4)

        # level 1 of unet
        [level_us3, crop0] = self.level_design(conc23, 'level_us3', filters1=25, filters2=32,
                                               is_training=is_training, kernel_size=1, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')

        conv1 = tf.layers.conv3d(level_us3,
                                 filters=32,
                                 kernel_size=1,
                                 padding='same',
                                 activation=None,
                                 dilation_rate=2,
                                 )
        bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
        bn = tf.nn.relu(bn)



        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        with tf.name_scope('classification_layer'):
            y = tf.layers.conv3d(bn, filters=self.class_no, kernel_size=14, padding='same', strides=(1, 1, 1),
                                 activation=None, dilation_rate=(1, 1,1), name='fc3'+self.log_ext)

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y#,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6

