import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import os
from os import listdir
from os.path import isfile, join
# import matplotlib.pyplot as plt
import time
from functions.layers.upsampling import upsampling
from functions.tf_augmentation.augmentation import augmentation
# !!

class _densenet:
    def __init__(self, class_no=14):
        print('create object _unet')
        self.class_no = class_no
        self.kernel_size1 = 1
        self.kernel_size2 = 3
        self.log_ext = '_'
        self.seed_no=200
        self.upsampling3d=upsampling()

        self.maxpool=False # use maxpool or strided conv for downsampling
        self.norm_method = 'batch_normalization'
        self.augmentation=augmentation(self.seed_no)


    # ========================
    def normalization(self,input,norm_method,training,renorm=False):
        bn=None
        if norm_method=='batch_normalization':
            bn = tf.layers.batch_normalization(input, training=training, renorm=renorm)
        elif norm_method=='instance_norm':
            bn = tf.contrib.layers.instance_norm(
                input,
                center=True,
                scale=True,
                epsilon=1e-06,
                activation_fn=None,
                param_initializers=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                data_format= 'NHWC',
                scope=None
            )
        return bn

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
            bn = self.normalization(conv1,norm_method=self.norm_method, training=is_training, renorm=False)
            bn = tf.nn.leaky_relu(bn)
            conv1 = bn


            conc=tf.concat([input[:,1:-1,1:-1,1:-1,:],conv1],4)

            return conc
    #==================================================
    def conv3d_layer(self,input,filters,kernel_size,padding,dilation_rate,is_training,strides):
        conv = tf.layers.conv3d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding=padding,
                                 activation=None,
                                 dilation_rate=dilation_rate,
                                strides=strides
                                 )
        bn = self.normalization(conv,norm_method=self.norm_method, training=is_training, renorm=False)
        bn = tf.nn.leaky_relu(bn)
        conv = bn
        return conv
    #==================================================
    def level_design(self,input,level_name,filters1,filters2,is_training,kernel_size,in_size,crop_size,padding1,padding2,flag=2,paddingfree_scope='',filters3=0):
        with tf.variable_scope(level_name):
            conv1 =self.conv3d_layer( input, filters=filters1, kernel_size=kernel_size, padding=padding1, dilation_rate=1, is_training=is_training,strides=1)

            conc = tf.concat([input, conv1], 4)

            conv2 = self.conv3d_layer(conc, filters=filters2, kernel_size=kernel_size, padding=padding2,
                                      dilation_rate=1, is_training=is_training,strides=1)


            conc=tf.concat([conv1,conv2],4)


            # bottleneck layer
            conv3 = tf.layers.conv3d(conc,
                                     filters=filters2,
                                     kernel_size=1,
                                     padding=padding2,
                                     activation=None,
                                     dilation_rate=1,
                                     )
            bn = self.normalization(conv3, norm_method=self.norm_method,training=is_training, renorm=False)
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




    #==================================================
    def paddingfree_conv(self,input,filters,kernel_size,is_training):
        conv = tf.layers.conv3d(input,
                                 filters=filters,
                                 kernel_size=kernel_size,
                                 padding='valid',
                                 activation=None,
                                 dilation_rate=1,
                                 )
        bn = self.normalization(conv,norm_method=self.norm_method, training=is_training, renorm=False)
        bn = tf.nn.leaky_relu(bn)
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
    def densenet(self, img_row1, img_row2, img_row3, img_row4, img_row5, img_row6, img_row7, img_row8, input_dim, is_training,mri=None,conv_transpose=False):
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
                img_rows=self.augmentation.noisy_input( img_rows,is_training)
            with tf.variable_scope('LR_flip'):
                img_rows=self.augmentation.flip_lr_input(img_rows, is_training)
            with tf.variable_scope('rotate'):
                img_rows,degree=self.augmentation.rotate_input(img_rows, is_training)

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
                                               filters1=8,
                                               filters2=8,
                                               filters3=8,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv1',
                                               flag=1)
        if self.maxpool:
            with tf.variable_scope('maxpool1'):
                pool1 = tf.layers.max_pooling3d(inputs=level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        else:
            with tf.variable_scope('strided_ds1'):
                pool1 = self.conv3d_layer( input=level_ds1, filters=level_ds1.shape[-1], kernel_size=3, padding='valid', dilation_rate=1, is_training=is_training,strides=2)
        # level 2 of unet
        [level_ds2, crop2] = self.level_design(pool1,
                                               'level_ds2',
                                               filters1=16,
                                               filters2=16,
                                               filters3=16,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               paddingfree_scope='paddingfree_conv2',
                                               flag=1)
        if self.maxpool:
            with tf.variable_scope('maxpool2'):
                pool2 = tf.layers.max_pooling3d(inputs=level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))
        else:
            with tf.variable_scope('strided_ds2'):
                pool2 = self.conv3d_layer( input=level_ds2, filters=level_ds2.shape[-1], kernel_size=3, padding='valid', dilation_rate=1, is_training=is_training,strides=2)

        # level 3 of unet
        [level_ds3, crop3] = self.level_design(pool2, 'level_ds3',
                                               filters1=24,
                                               filters2=24,
                                               filters3=24,
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
                                                     filters=24,
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
                                               filters1=16,
                                               filters2=16,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same', padding2='same')
        if conv_transpose:
            with tf.variable_scope('conv_transpose2'):
                deconv2 = tf.layers.conv3d_transpose(level_us2,
                                                     filters=16,
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
            y=conv1



        # == == == == == == == == == == == == == == == == == == == == == ==


        # classification layer:
        # with tf.variable_scope('classification_layer'):
        #     y = tf.layers.conv3d(bn, filters=self.class_no, kernel_size=14, padding='same', strides=(1, 1, 1),
        #                          activation=None, dilation_rate=(1, 1,1), name='fc3'+self.log_ext)

        print(' total number of variables %s' % (
            np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))





        return  y,degree

