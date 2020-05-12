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

class _forked_densenet:
    def __init__(self, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed=200
        self.upsampling3d=upsampling()

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
            bn = tf.nn.relu(bn)
            conv1 = bn

            # conv2 = tf.layers.conv3d(conv1,
            #                          filters=filters,
            #                          kernel_size=kernel_size,
            #                          padding=padding,
            #                          activation=None,
            #                          dilation_rate=1)
            # bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            # bn = tf.nn.relu(bn)
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
            # bn = tf.nn.relu(bn)
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
            bn = tf.nn.relu(bn)
            conv1 = bn

            conv2 = tf.layers.conv3d(conv1,
                                     filters=filters2,
                                     kernel_size=kernel_size,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv2, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)
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
            bn = tf.nn.relu(bn)
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
                # if np.shape(conc)[1]!=in_size:
                #     print("pppppppp")
            if flag==2:
                crop = conc[:,
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                       tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                       tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            # crop=tf.cond(tf.equal(in_size,crop_size), lambda : conv3,lambda :crop)


            return conc,crop

    #==================================================
    def noisy_input(self,img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,is_training):
        with tf.variable_scope("Noise"):
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

    def rotate_input(self,img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8,
                    is_training):
        with tf.variable_scope("Rotate"):
            with tf.Session() as s:
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
        with tf.variable_scope("LR_Flip"):
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
    def paddingfree_conv(self,input,filters,kernel_size,is_training):
        conv = tf.layers.conv3d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding='valid',
                                 activation=None,
                                 dilation_rate=1,
                                 )
        bn = tf.layers.batch_normalization(conv, training=is_training, renorm=False)
        bn = tf.nn.relu(bn)
        conv = bn
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
        in_size0 = tf.to_int32(0)
        in_size1 = tf.to_int32(input_dim)
        in_size2 = tf.to_int32(in_size1 )  # conv stack
        in_size3 = tf.to_int32((in_size2-2))  # level_design1
        in_size4 = tf.to_int32(in_size3 / 2-2 )  # downsampleing1+level_design2
        in_size5 = tf.to_int32(in_size4 / 2 -2)  # downsampleing2+level_design3
        # in_size6 = int(in_size5 / 2 -2)  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0)
        crop_size1 = tf.to_int32(2 * in_size5 + 1)
        crop_size2 = tf.to_int32(2 * crop_size1  + 1)
        # crop_size3 = (2 * crop_size2  + 1)
        # final_layer = crop_size3
        with tf.variable_scope('augmentation'):
            with tf.variable_scope('noise'):
                [img_row1, img_row2, img_row3, img_row4,
                 img_row5, img_row6, img_row7, img_row8]=self.noisy_input( img_row1, img_row2, img_row3, img_row4,
                                                                           img_row5, img_row6, img_row7, img_row8,
                                                                           is_training)
            with tf.variable_scope('LR_flip'):
                [img_row1, img_row2, img_row3, img_row4,
                 img_row5, img_row6, img_row7, img_row8]=self.flip_lr_input(img_row1, img_row2, img_row3, img_row4,
                                                                            img_row5, img_row6, img_row7, img_row8,
                                                                            is_training)
        filters=4
        # with tf.variable_scope('stack-layers'):
            # stack1=self.convolution_stack(input=img_row1, stack_name='stack_1', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack2=self.convolution_stack(input=img_row2, stack_name='stack_2', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack3=self.convolution_stack(input=img_row3, stack_name='stack_3', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack4=self.convolution_stack(input=img_row4, stack_name='stack_4', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack5=self.convolution_stack(input=img_row5, stack_name='stack_5', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack6=self.convolution_stack(input=img_row6, stack_name='stack_6', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack7=self.convolution_stack(input=img_row7, stack_name='stack_7', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
            # stack8=self.convolution_stack(input=img_row8, stack_name='stack_8', filters=filters, kernel_size=3, padding='valid', is_training=is_training)
        with tf.variable_scope('stack-contact'):
            stack_concat = tf.concat([img_row1, img_row2], 4)
            stack_concat = tf.concat([stack_concat, img_row3], 4)
            stack_concat = tf.concat([stack_concat, img_row4], 4)
            stack_concat = tf.concat([stack_concat, img_row5], 4)
            stack_concat = tf.concat([stack_concat, img_row6], 4)
            stack_concat = tf.concat([stack_concat, img_row7], 4)
            stack_concat = tf.concat([stack_concat, img_row8], 4)
        #== == == == == == == == == == == == == == == == == == == == == ==
        #level 1 of unet
        [level_ds1, crop1] = self.level_design(stack_concat,
                                               'level_ds1',
                                               filters1=14,
                                               filters2=14,
                                               filters3=14,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv1',
                                               flag=1)
        # with tf.variable_scope():
        #     level_ds1=self.paddingfree_conv( input=level_ds1, filters=24, kernel_size=3, is_training=is_training)
        with tf.variable_scope('maxpool1'):
            pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1,
                                               'level_ds2',
                                               filters1=28,
                                               filters2=28,
                                               filters3=28,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv2',
                                               flag=1)
        # with tf.variable_scope('paddingfree_conv2'):
        #     level_ds2 = self.paddingfree_conv(input=level_ds2, filters=45, kernel_size=3, is_training=is_training)
        with tf.variable_scope('maxpool2'):
            pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [level_ds3, crop3] = self.level_design(pool2, 'level_ds3',
                                               filters1=42,
                                               filters2=42,
                                               filters3=42,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size5,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv3',
                                               flag=1)
        # with tf.variable_scope('paddingfree_conv3'):
        #     level_ds3 = self.paddingfree_conv(input=level_ds3, filters=45, kernel_size=3, is_training=is_training)

        # == == == == == == == == == == == == == == == == == == == == == ==
        # with tf.variable_scope('maxpool3'):
        #     pool3 = tf.layers.max_pooling3d(inputs=level_ds3, pool_size=(2, 2, 2), strides=(2, 2, 2))
        #
        # # level 3 of unet
        # [level_us1, crop0] = self.level_design(pool3, 'level_us1',
        #                                        filters1=56,
        #                                        filters2=56,
        #                                        filters3=56,
        #                                        is_training=is_training,
        #                                        kernel_size=3,
        #                                        in_size=in_size6,
        #                                        crop_size=crop_size0,
        #                                        padding1='same',
        #                                        padding2='same',
        #                                        paddingfree_scope='paddingfree_conv4',
        #                                        flag=1)
        # with tf.variable_scope('paddingfree_conv4'):
        #     level_us1 = self.paddingfree_conv(input=level_us1, filters=64, kernel_size=3, is_training=is_training)

        # == == == == == == == == == == == == == == == == == == == == == ==


        with tf.variable_scope('conv_transpose1'):
            # deconv1=self.upsampling3d.upsampling3d(level_us1, 'scope1')
            deconv1 = tf.layers.conv3d_transpose(level_ds3,
                                                 filters=28,
                                                 kernel_size=3,
                                                 strides=(2, 2, 2),
                                                 padding='valid',
                                                 use_bias=False)
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

        with tf.variable_scope('conv_transpose2'):
            # deconv2 = self.upsampling3d.upsampling3d(level_us2, 'scope2')
            deconv2 = tf.layers.conv3d_transpose(level_us2,
                                                 filters=28,
                                                 kernel_size=3,
                                                 strides=(2, 2, 2),
                                                 padding='valid',
                                                 use_bias=False)
        with tf.variable_scope('concat2'):
            conc23 = tf.concat([crop1, deconv2], 4)

        # level 1 of unet
        [level_us3, crop0] = self.level_design(conc23, 'level_us3',
                                               filters1=14,
                                               filters2=14,
                                               is_training=is_training,
                                               kernel_size=1,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        # with tf.variable_scope('conv_transpose3'):
        #     # deconv3 = self.upsampling3d.upsampling3d(level_us3, 'scope3')
        #     deconv3 = tf.layers.conv3d_transpose(level_us3,
        #                                          filters=14,
        #                                          kernel_size=3,
        #                                          strides=(2, 2, 2),
        #                                          padding='valid',
        #                                          use_bias=False)
        # with tf.variable_scope('concat3'):
        #     conc34 = tf.concat([crop1, deconv3], 4)
        #
        # # level 1 of unet
        # [level_us4, crop0] = self.level_design(conc34, 'level_us4',
        #                                        filters1=14,
        #                                        filters2=14,
        #                                        is_training=is_training,
        #                                        kernel_size=1,
        #                                        in_size=in_size0,
        #                                        crop_size=crop_size0,
        #                                        padding1='same',
        #                                        padding2='same')
        with tf.variable_scope('last_conv'):
            conv1 = tf.layers.conv3d(level_us3,
                                     filters=14,
                                     kernel_size=3,
                                     padding='same',
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = tf.layers.batch_normalization(conv1, training=is_training, renorm=False)
            bn = tf.nn.relu(bn)



        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        with tf.variable_scope('last_layer'):
            y = tf.layers.conv3d(bn, filters=14, kernel_size=1, padding='same', strides=(1, 1, 1),
                                 activation=None, dilation_rate=(1, 1,1), name='lastconv')

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y#,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6

