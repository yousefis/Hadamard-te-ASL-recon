import math
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
from functions.layers.conv_kernel import *
from functions.tf_augmentation.augmentation import augmentation
# !!

class _reverse_densenet:
    def __init__(self, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed_no=200
        self.tf_augmentation=augmentation(self.seed_no)

        self.upsampling3d=upsampling()
    def seed(self):
        self.seed_no+=1
        return self.seed_no

    # ========================
    def convolution_stack(self, input, stack_name, filters, kernel_size, padding, is_training):
        with tf.variable_scope(stack_name):
            conv1 = tf.layers.conv3d(input,
                                        filters=filters,
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        activation=None,
                                        dilation_rate=1,
                                        )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv1 = bn

            # conv2 = tf.layers.conv3d(conv1,
            #                          filters=filters,
            #                          kernel_size=kernel_size,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            # bn = tf.nn.leaky_relu(bn)
            # conv2 = bn
            conc=tf.concat([input[:,1:-1,1:-1,1:-1,:],conv1],4)
            # conc=tf.concat([conc,conv2],4)

            # conv3 = tf.layers.conv3d(conc,
            #                          filters=filters,
            #                          kernel_size=1,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            # bn = tf.nn.leaky_relu(bn)
            # conv3 = bn
            return conc
    #==================================================
    def level_design(self,input,level_name,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2,flag=2,paddingfree_scope='',filters3=0):
        with tf.variable_scope(level_name):
            conv1 = tf.layers.conv3d(input,
                                     filters=filters1,
                                     kernel_size=kernel_size,
                                     padding=padding1,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv1 = bn
            conc = tf.concat([input, conv1], 4)

            conv2 = tf.layers.conv3d(conc,
                                     filters=filters2,
                                     kernel_size=kernel_size,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv2 = bn


            conc=tf.concat([conv1,conv2],4)


            # bottleneck layer
            conv3 = tf.layers.conv3d(conc,
                                     filters=filters2,
                                     kernel_size=1,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv3 = bn
            if flag==1:
                with tf.variable_scope(paddingfree_scope):
                    conc = self.paddingfree_conv(input=conv3, filters=filters3, kernel_size=3, is_training=is_training)
                crop = conc[:,
                          tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                          tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                          tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                          tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                          tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                          tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]
            if flag==2:
                crop = conc[:,
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            return conc,crop





    #==================================================
    def paddingfree_conv(self,input,filters,kernel_size,is_training):
        conv = tf.layers.conv3d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding='valid',
                                 activation=None,
                                 dilation_rate=1,
                                 )
        bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
        bn = tf.nn.leaky_relu(bn)
        conv = bn
        return conv
    #==================================================




    #==================================================
    def reverse_densenet(self, img_row1, img_row2, img_row3, img_row4, img_row5,
                         img_row6, img_row7, img_row8,
                         input_dim, is_training,mri=None,conv_transpose=False):
        mri=None
        # in_size0 = tf.to_int32(0)
        # in_size1 = tf.to_int32(input_dim)
        # in_size2 = tf.to_int32(in_size1)  # conv stack
        # in_size3 = tf.to_int32((in_size2 - 2))  # level_design1
        # in_size4 = tf.to_int32(in_size3 / 2 - 2)  # downsampleing1+level_design2
        # in_size5 = tf.to_int32(in_size4 / 2 - 2)  # downsampleing2+level_design3
        # in_size6 = tf.to_int32(in_size5 / 2 - 2)  # downsampleing2+level_design3
        # crop_size0 = tf.to_int32(0)
        # crop_size1 = tf.to_int32(2 * in_size6 + 1)
        # crop_size2 = tf.to_int32(2 * crop_size1 + 1)
        # crop_size3 = tf.to_int32(2 * crop_size2 + 1)

        # input_dim=77
        with tf.variable_scope('crop_claculation'):
            in_size0 = tf.to_int32(0,name='in_size0')
            in_size1 = tf.to_int32(input_dim,name='in_size1')
            in_size2 = tf.to_int32(in_size1 ,name='in_size2')  # conv stack
            in_size3 = tf.to_int32((in_size2-2),name='in_size3')  # level_design1
            in_size4 = tf.to_int32(in_size3 / 2-2 ,name='in_size4')  # downsampleing1+level_design2
            in_size5 = tf.to_int32(in_size4 / 2 -2,name='in_size5')  # downsampleing2+level_design3
            crop_size0 = tf.to_int32(0,name='crop_size0')
            crop_size1 = tf.to_int32(2 * in_size5 + 1,name='crop_size1')
            crop_size2 = tf.to_int32(2 * crop_size1  + 1,name='crop_size2')

        img_rows=[]
        img_rows.append(img_row1)
        img_rows.append(img_row2)
        img_rows.append(img_row3)
        img_rows.append(img_row4)
        img_rows.append(img_row5)
        img_rows.append(img_row6)
        img_rows.append(img_row7)
        img_rows.append(img_row8)
        if mri !=None:
            img_rows.append(mri)

        with tf.variable_scope('augmentation'):
            with tf.variable_scope('noise'):
                img_rows=self.tf_augmentation.noisy_input( img_rows,is_training)
            with tf.variable_scope('LR_flip'):
                img_rows=self.tf_augmentation.flip_lr_input(img_rows, is_training)
            with tf.variable_scope('rotate'):
                img_rows,degree=self.tf_augmentation.rotate_input(img_rows, is_training)

        with tf.variable_scope('stack-contact'):
            for i in range(len(img_rows)):
                if i==0:
                    stack_concat = tf.concat([img_rows[0], img_rows[1]], 4)
                elif i == 1:
                    continue
                else:
                    stack_concat = tf.concat([stack_concat, img_rows[i]], 4)

            # stack_concat = tf.concat([stack_concat, img_rows[3]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[4]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[5]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[6]], 4)
            # stack_concat = tf.concat([stack_concat, img_rows[7]], 4)
        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack_concat,
                                               'level_ds1',
                                               filters1=10,
                                               filters2=10,
                                               filters3=10,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv1',
                                               flag=1)
        with tf.variable_scope('maxpool1'):
            pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1,
                                               'level_ds2',
                                               filters1=25,
                                               filters2=25,
                                               filters3=25,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv2',
                                               flag=1)
        with tf.variable_scope('maxpool2'):
            pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [level_ds3, crop3] = self.level_design(pool2, 'level_ds3',
                                               filters1=32,
                                               filters2=32,
                                               filters3=32,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size5,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv3',
                                               flag=1)

        if conv_transpose:
            with tf.variable_scope('conv_transpose1'):
                deconv1 = tf.layers.conv3d_transpose(level_ds3,
                                                     filters=28,
                                                     kernel_size=3,
                                                     strides=(2, 2, 2),
                                                     padding='valid',
                                                     use_bias=False)
        else:
            with tf.variable_scope('upsampling1'):
                deconv1 = self.upsampling3d.upsampling3d(level_ds3, scope='upsampling1', scale=2,
                                           interpolator='trilinear', padding_mode='SYMMETRIC',
                                           padding_constant=0, trainable=False,
                                           padding='valid')
        with tf.variable_scope('concat1'):
            conc12 = tf.concat([crop2, deconv1], 4)

        # level 2 of unet
        [level_us2, crop0] = self.level_design(conc12, 'level_us2',
                                               filters1=28,
                                               filters2=28,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        if conv_transpose:
            with tf.variable_scope('conv_transpose2'):
                deconv2 = tf.layers.conv3d_transpose(level_us2,
                                                     filters=14,
                                                     kernel_size=3,
                                                     strides=(2, 2, 2),
                                                     padding='valid',
                                                     use_bias=False)
        else:
            with tf.variable_scope('upsampling2'):
                deconv2 = self.upsampling3d.upsampling3d(level_us2, scope='upsampling2', scale=2,
                                           interpolator='trilinear', padding_mode='SYMMETRIC',
                                           padding_constant=0, trainable=False,
                                           padding='valid')
        with tf.variable_scope('concat2'):
            conc23 = tf.concat([crop1, deconv2], 4)

        # level 1 of unet
        [level_us3, crop0] = self.level_design(conc23, 'level_us3',
                                               filters1=14,
                                               filters2=14,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')

        with tf.variable_scope('last_layer'):
            conv1 = tf.layers.conv3d(level_us3,
                                     filters=14,
                                     kernel_size=1,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            # bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            # y = tf.nn.leaky_relu(bn)
            y=conv1



        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        # with tf.variable_scope('classification_layer'):
        #     y = tf.layers.conv3d(bn, filters=self.class_no, kernel_size=14, padding='same', strides=(1, 1, 1),
        #                          activation=None, dilation_rate=(1, 1,1), name='fc3'+self.log_ext)

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y,degree

