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

class _densenet:
    def __init__(self, graph,class_no=14):
        print('create object _unet')
        self.graph=graph
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed=200
        self.upsampling3d=upsampling()
        self.layers = layers()

        self.reuse=False
        self.trainable=True
    # ========================
    def convolution_stack(self, input, stack_name, filters, kernel_size, padding, is_training):
        with tf.variable_scope(stack_name):
            conv1 = self.layers.conv3d(input,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=stack_name + '_conv1',
                                       reuse=self.reuse)

            conv2 = self.layers.conv3d(conv1,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=stack_name + '_conv2',
                                       reuse=self.reuse)

            conc=tf.concat([input,conv1],4)
            conc=tf.concat([conc,conv2],4)

            conv3 = self.layers.conv3d(conc,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding=padding,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=stack_name + '_conv3',
                                       reuse=self.reuse)
            return conv3
    #==================================================
    def level_design(self,input,level_name,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2):
        with tf.variable_scope(level_name):
            conv1 = self.layers.conv3d(input,
                                       filters=filters1,
                                       kernel_size=kernel_size,
                                       padding=padding1,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=level_name + '_conv1',
                                       reuse=self.reuse)
            conv2 = self.layers.conv3d(conv1,
                                       filters=filters2,
                                       kernel_size=kernel_size,
                                       padding=padding2,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=level_name + '_conv2',
                                       reuse=self.reuse)

            conc=tf.concat([conv1,conv2],4,name=level_name+'_concat')

            #bottleneck layer
            # conv3 = tf.layers.conv3d(conc,
            #                          filters=filters2/2,
            #                          kernel_size=1,
            #                          padding=padding2,
            #                          activation=None,
            #                          dilation_rate=1,
            #                          )
            # bn = tf.layers.batch_normalization(conv3, training=is_training, renorm=False)
            # bn = tf.nn.relu(bn)
            # conv3 = bn

            crop = conc[:,
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                      tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                      tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            # crop=tf.cond(tf.equal(in_size,crop_size), lambda : conv3,lambda :crop)


            return conv2,crop

    #==================================================
    def noisy_input(self,img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,is_training):
        with tf.Session(graph=self.graph) as s:
            rnd = s.run(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed))  # , seed=int(time.time())))

        noisy_img_row1 = tf.cond(is_training,
                                 lambda: img_row1 + tf.round(tf.random_normal(tf.shape(img_row1), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row1,name='noisy_img_row1')
        noisy_img_row2 = tf.cond(is_training,
                                 lambda: img_row2 + tf.round(tf.random_normal(tf.shape(img_row2), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row2,name='noisy_img_row2')
        noisy_img_row3 = tf.cond(is_training,
                                 lambda: img_row3 + tf.round(tf.random_normal(tf.shape(img_row3), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row3,name='noisy_img_row3')
        noisy_img_row4 = tf.cond(is_training,
                                 lambda: img_row4 + tf.round(tf.random_normal(tf.shape(img_row4), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row4,name='noisy_img_row4')
        noisy_img_row5 = tf.cond(is_training,
                                 lambda: img_row5 + tf.round(tf.random_normal(tf.shape(img_row5), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row5,name='noisy_img_row5')
        noisy_img_row6 = tf.cond(is_training,
                                 lambda: img_row6 + tf.round(tf.random_normal(tf.shape(img_row6), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row6,name='noisy_img_row6')
        noisy_img_row7 = tf.cond(is_training,
                                 lambda: img_row7 + tf.round(tf.random_normal(tf.shape(img_row7), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row7,name='noisy_img_row7')
        noisy_img_row8 = tf.cond(is_training,
                                 lambda: img_row8 + tf.round(tf.random_normal(tf.shape(img_row8), mean=0,
                                                                              stddev=rnd,
                                                                              seed=self.seed + 2,  # int(time.time()),
                                                                              dtype=tf.float32))
                                 , lambda: img_row8,name='noisy_img_row8')
        return noisy_img_row1,noisy_img_row2,noisy_img_row3,noisy_img_row4,\
               noisy_img_row5,noisy_img_row6,noisy_img_row7,noisy_img_row8

    # ==================================================

    def rotate_input(self,img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,
                    is_training):
        with tf.Session(graph=self.graph) as s:
            rnd = s.run(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed),5))  # , seed=int(time.time())))


        rotate_img_row1 = tf.cond( tf.logical_and(is_training,rnd),
                                 lambda:  tf.expand_dims(tf.contrib.image.rotate(tf.squeeze(img_row1,4),),axis=4)
                                 , lambda: img_row1)
        rotate_img_row2 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row2,4)),axis=4)
                                 , lambda: img_row2)
        rotate_img_row3 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row3,4)),axis=4)
                                 , lambda: img_row3)
        rotate_img_row4 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row4,4)),axis=4)
                                 , lambda: img_row4)
        rotate_img_row5 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row5,4)),axis=4)
                                 , lambda: img_row5)
        rotate_img_row6 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row6,4)),axis=4)
                                 , lambda: img_row6)
        rotate_img_row7 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row7,4)),axis=4)
                                 , lambda: img_row7)
        rotate_img_row8 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row8,4)),axis=4)
                                 , lambda: img_row8)
        return rotate_img_row1, rotate_img_row2, rotate_img_row3, rotate_img_row4, \
               rotate_img_row5, rotate_img_row6, rotate_img_row7, rotate_img_row8
    # ========================

    def flip_lr_input(self, img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,
                    is_training):
        with tf.Session(graph=self.graph) as s:
            rnd = s.run(tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed),5))  # , seed=int(time.time())))

        flip_lr_img_row1 = tf.cond( tf.logical_and(is_training,rnd),
                                 lambda:  tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row1,4)),axis=4)
                                 , lambda: img_row1,name='flip_lr_img_row1')
        flip_lr_img_row2 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row2,4)),axis=4)
                                 , lambda: img_row2,name='flip_lr_img_row2')
        flip_lr_img_row3 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row3,4)),axis=4)
                                 , lambda: img_row3,name='flip_lr_img_row3')
        flip_lr_img_row4 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row4,4)),axis=4)
                                 , lambda: img_row4,name='flip_lr_img_row4')
        flip_lr_img_row5 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row5,4)),axis=4)
                                 , lambda: img_row5,name='flip_lr_img_row5')
        flip_lr_img_row6 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row6,4)),axis=4)
                                 , lambda: img_row6,name='flip_lr_img_row6')
        flip_lr_img_row7 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda: tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row7,4)),axis=4)
                                 , lambda: img_row7,name='flip_lr_img_row7')
        flip_lr_img_row8 = tf.cond(tf.logical_and(is_training,rnd),
                                 lambda:tf.expand_dims(tf.image.flip_left_right(tf.squeeze(img_row8,4)),axis=4)
                                 , lambda: img_row8,name='flip_lr_img_row8')
        return flip_lr_img_row1, flip_lr_img_row2, flip_lr_img_row3, flip_lr_img_row4, \
               flip_lr_img_row5, flip_lr_img_row6, flip_lr_img_row7, flip_lr_img_row8
    #==================================================
    def paddingfree_conv(self,input,filters,kernel_size,is_training,name):
        with tf.variable_scope(name):
            conv = self.layers.conv3d(input,
                                       filters=filters,
                                       kernel_size=kernel_size,
                                       padding='valid',
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=name,
                                       reuse=self.reuse)
        return conv
    #==================================================
    def bspline3D(self,x, y, z):
        x = abs(x)
        y = abs(y)
        z = abs(z)
        if (((x + y + z) >= 0) and ((x + y + z) < 1)):
            f = (0.5) * ((x + y + z) ** 3) - ((x + y + z) ** 2) + 4 / 6
        elif ((x + y + z) >= 1 and (x + y + z) <= 2):
            f = (-1 / 6) * ((x + y + z) ** 3) + ((x + y + z) ** 2) - 2 * (x + y + z) + 8 / 6
        else:
            f = 0
        return f

    #==================================================
    def convDownsampleKernel(self,kernelName, dimension, kernelSize, normalizeKernel=None):
        numOfPoints = kernelSize + 2
        XInput = np.linspace(-2, 2, num=numOfPoints)

        if dimension == 3:
            YInput = np.linspace(-2, 2, num=numOfPoints)
            ZInput = np.linspace(-2, 2, num=numOfPoints)
            xv, yv, zv = np.meshgrid(XInput, YInput, ZInput)

            if kernelName == 'bspline':
                Y = np.stack(
                    [self.bspline3D(xv[i, j, k], yv[i, j, k], zv[i, j, k]) for i in range(0, np.shape(xv)[0]) for j in
                     range(0, np.shape(xv)[0])
                     for k in range(0, np.shape(xv)[0])])
            Y = np.reshape(Y, [len(XInput), len(XInput), len(XInput)])
            Y = Y[1:-1, 1:-1, 1:-1]
        if normalizeKernel:
            if np.sum(Y) != normalizeKernel:
                ratio = normalizeKernel / np.sum(Y)
                Y = ratio * Y

        Y[abs(Y) < 1e-6] = 0
        return Y.astype(np.float32)




    #==================================================
    def densenet(self, img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8, input_dim, is_training):


        in_size0 = tf.to_int32(0,name='in_size0')
        in_size1 = tf.to_int32(input_dim,name='in_size1')
        in_size2 = tf.to_int32(in_size1,name='in_size2')  # conv stack
        in_size3 = tf.to_int32((in_size2-2),name='in_size3')  # level_design1
        in_size4 = tf.to_int32(in_size3 / 2-2,name='in_size4')  # downsampleing1+level_design2
        in_size5 = tf.to_int32(in_size4 / 2-2 ,name='in_size5')  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0,name='crop_size0')
        crop_size1 = tf.to_int32(2 * in_size5 + 1,name='crop_size1')
        crop_size2 = tf.to_int32(2 * (crop_size1 ) + 1,name='crop_size2')

        # input_dim=53
        # in_size0 = (0)
        # in_size1 = (input_dim)
        # in_size2 = (in_size1 )  # conv stack
        # in_size3 = ((in_size2-2))  # level_design1
        # in_size4 = int(in_size3 / 2-2 )  # downsampleing1+level_design2
        # in_size5 = int(in_size4 / 2 -2)  # downsampleing2+level_design3
        # crop_size0 = (0)
        # crop_size1 = (2 * in_size5 + 1)
        # crop_size2 = (2 * (crop_size1 ) + 1)
        # final_layer = crop_size2-2*2
        with tf.variable_scope('S_add_noise'):
            [img_row1, img_row2, img_row3, img_row4,
             img_row5, img_row6, img_row7, img_row8]=self.noisy_input( img_row1, img_row2, img_row3, img_row4,
                                                                       img_row5, img_row6, img_row7, img_row8,
                                                                       is_training)
        with tf.variable_scope('S_flip_lr'):
            [img_row1, img_row2, img_row3, img_row4,
             img_row5, img_row6, img_row7, img_row8]=self.flip_lr_input(img_row1, img_row2, img_row3, img_row4,
                                                                        img_row5, img_row6, img_row7, img_row8,
                                                                        is_training)
        filters=8
        with tf.variable_scope('S_concat_stack'):
            stack1=self.convolution_stack(input=img_row1, stack_name='synth_stack_1', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack2=self.convolution_stack(input=img_row2, stack_name='synth_stack_2', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack3=self.convolution_stack(input=img_row3, stack_name='synth_stack_3', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack4=self.convolution_stack(input=img_row4, stack_name='synth_stack_4', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack5=self.convolution_stack(input=img_row5, stack_name='synth_stack_5', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack6=self.convolution_stack(input=img_row6, stack_name='synth_stack_6', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack7=self.convolution_stack(input=img_row7, stack_name='synth_stack_7', filters=filters, kernel_size=3, padding='same', is_training=is_training)
            stack8=self.convolution_stack(input=img_row8, stack_name='synth_stack_8', filters=filters, kernel_size=3, padding='same', is_training=is_training)

            stack_concat = tf.concat([stack1, stack2], 4)
            stack_concat = tf.concat([stack_concat, stack3], 4)
            stack_concat = tf.concat([stack_concat, stack4], 4)
            stack_concat = tf.concat([stack_concat, stack5], 4)
            stack_concat = tf.concat([stack_concat, stack6], 4)
            stack_concat = tf.concat([stack_concat, stack7], 4)
            stack_concat = tf.concat([stack_concat, stack8], 4,name='stack_concat')
        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack_concat, 'S_DS1', filters1=8, filters2=16,
                                               is_training=is_training, kernel_size=3, in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',padding2='same')

        level_ds1=self.paddingfree_conv( input=level_ds1, filters=8, kernel_size=3, is_training=is_training,name='S_conv1')
        with tf.variable_scope('S_maxpool1'):
            pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1, 'S_DS2', filters1=25, filters2=32,
                                               is_training=is_training, kernel_size=3, in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same', padding2='same')
        level_ds2 = self.paddingfree_conv(input=level_ds2, filters=32, kernel_size=3, is_training=is_training,name='S_conv2')
        with tf.variable_scope('S_maxpool2'):
            pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [level_us1, crop0] = self.level_design(pool2, 'S_US1', filters1=32, filters2=40,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        level_us1 = self.paddingfree_conv(input=level_us1, filters=40, kernel_size=3, is_training=is_training,name='S_conv3')
        with tf.variable_scope('S_concat_upsampling1'):
            deconv1=self.upsampling3d.upsampling3d(level_us1, 'upsampling1')
            conc12 = tf.concat([crop2, deconv1], 4)

        # level 2 of unet
        [level_us2, crop0] = self.level_design(conc12, 'S_US12', filters1=45, filters2=32,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        with tf.variable_scope('S_concat_upsampling2'):
            deconv2 = self.upsampling3d.upsampling3d(level_us2, 'upsampling2')
            conc23 = tf.concat([crop1, deconv2], 4)

        # level 1 of unet
        [level_us3, crop0] = self.level_design(conc23, 'S_US3', filters1=25, filters2=32,
                                               is_training=is_training, kernel_size=1, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='same')
        with tf.variable_scope('S_conv'):
            conv1 = self.layers.conv3d(level_us3,
                                      filters=32,
                                      kernel_size=1,
                                      padding='same',
                                      dilation_rate=1,
                                      is_training=is_training,
                                      trainable=self.trainable,
                                      scope='conv',
                                      reuse=self.reuse)
        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        with tf.variable_scope('S_y'):
            y = tf.layers.conv3d(conv1, filters=self.class_no, kernel_size=14, padding='same', strides=(1, 1, 1),
                                 activation=None, dilation_rate=(1, 1,1),name='y')

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y#,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6

