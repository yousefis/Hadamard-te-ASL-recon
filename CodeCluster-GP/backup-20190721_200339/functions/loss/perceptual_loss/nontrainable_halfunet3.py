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

from functions.kernel_loader.loader import loader

# !!

class _unet:
    def __init__(self, trainable,file_name):
        print('create object _unet')
        self.upsampling3d = upsampling()
        self.layers = layers()
        self.trainable = trainable

        self.seed=200


        self.kernel_loader = loader(file_name)
        # self.kernel_loader.print_tensors_in_checkpoint_file(file_name, False,
        #                                                     True, False)
        [self.conv_init1_ld1,
        self.bias_init1_ld1,
        self.beta_init1_ld1,
        self.gamma_init1_ld1,
        self.moving_mean_init1_ld1,
        self.moving_var1_ld1,

        self.conv_init2_ld1,
        self.bias_init2_ld1,
        self.beta_init2_ld1,
        self.gamma_init2_ld1,
        self.moving_mean_init2_ld1,
        self.moving_var2_ld1,

        self.conv_init1_ld2,
        self.bias_init1_ld2,
        self.beta_init1_ld2,
        self.gamma_init1_ld2,
        self.moving_mean_init1_ld2,
        self.moving_var1_ld2,

        self.conv_init2_ld2,
        self.bias_init2_ld2,
        self.beta_init2_ld2,
        self.gamma_init2_ld2,
        self.moving_mean_init2_ld2,
        self.moving_var2_ld2,

        self.conv_init1_ld3,
        self.bias_init1_ld3,
        self.beta_init1_ld3,
        self.gamma_init1_ld3,
        self.moving_mean_init1_ld3,
        self.moving_var1_ld3,

        self.conv_init2_ld3,
        self.bias_init2_ld3,
        self.beta_init2_ld3,
        self.gamma_init2_ld3,
        self.moving_mean_init2_ld3,
        self.moving_var2_ld3]=self.kernel_loader.return_tensor_value_list_by_name(['U_LD_DS1/U_LD_DS1U_conv1_conv3d/kernel',
                                                             'U_LD_DS1/U_LD_DS1U_conv1_conv3d/bias',
                                                             'U_LD_DS1/U_LD_DS1U_conv1_bn/beta',
                                                             'U_LD_DS1/U_LD_DS1U_conv1_bn/gamma',
                                                             'U_LD_DS1/U_LD_DS1U_conv1_bn/moving_mean',
                                                             'U_LD_DS1/U_LD_DS1U_conv1_bn/moving_variance',

                                                             'U_LD_DS1/U_LD_DS1U_conv2_conv3d/kernel',
                                                             'U_LD_DS1/U_LD_DS1U_conv2_conv3d/bias',
                                                             'U_LD_DS1/U_LD_DS1U_conv2_bn/beta',
                                                             'U_LD_DS1/U_LD_DS1U_conv2_bn/gamma',
                                                             'U_LD_DS1/U_LD_DS1U_conv2_bn/moving_mean',
                                                             'U_LD_DS1/U_LD_DS1U_conv2_bn/moving_variance',

                                                             'U_LD_DS2/U_LD_DS2U_conv1_conv3d/kernel',
                                                             'U_LD_DS2/U_LD_DS2U_conv1_conv3d/bias',
                                                             'U_LD_DS2/U_LD_DS2U_conv1_bn/beta',
                                                             'U_LD_DS2/U_LD_DS2U_conv1_bn/gamma',
                                                             'U_LD_DS2/U_LD_DS2U_conv1_bn/moving_mean',
                                                             'U_LD_DS2/U_LD_DS2U_conv1_bn/moving_variance',

                                                             'U_LD_DS2/U_LD_DS2U_conv2_conv3d/kernel',
                                                             'U_LD_DS2/U_LD_DS2U_conv2_conv3d/bias',
                                                             'U_LD_DS2/U_LD_DS2U_conv2_bn/beta',
                                                             'U_LD_DS2/U_LD_DS2U_conv2_bn/gamma',
                                                             'U_LD_DS2/U_LD_DS2U_conv2_bn/moving_mean',
                                                             'U_LD_DS2/U_LD_DS2U_conv2_bn/moving_variance',

                                                             'U_LD_US1/U_LD_US1U_conv1_conv3d/kernel',
                                                             'U_LD_US1/U_LD_US1U_conv1_conv3d/bias',
                                                             'U_LD_US1/U_LD_US1U_conv1_bn/beta',
                                                             'U_LD_US1/U_LD_US1U_conv1_bn/gamma',
                                                             'U_LD_US1/U_LD_US1U_conv1_bn/moving_mean',
                                                             'U_LD_US1/U_LD_US1U_conv1_bn/moving_variance',

                                                             'U_LD_US1/U_LD_US1U_conv2_conv3d/kernel',
                                                             'U_LD_US1/U_LD_US1U_conv2_conv3d/bias',
                                                             'U_LD_US1/U_LD_US1U_conv2_bn/beta',
                                                             'U_LD_US1/U_LD_US1U_conv2_bn/gamma',
                                                             'U_LD_US1/U_LD_US1U_conv2_bn/moving_mean',
                                                             'U_LD_US1/U_LD_US1U_conv2_bn/moving_variance',

                                                             ])

    # ==================================================
    def level_design(self, input, filters1, filters2, is_training, kernel_size, in_size, crop_size, padding1, padding2,scope,
                     conv_init1, bias_init1, beta_init1, gamma_init1, moving_mean_init1, moving_var1,
                     conv_init2, bias_init2, beta_init2, gamma_init2, moving_mean_init2, moving_var2):
        with tf.variable_scope(scope):
            conv1 = self.layers.init_conv3d(input,
                                       filters=filters1,
                                       kernel_size=kernel_size,
                                       padding=padding1,
                                       dilation_rate=1,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=scope + 'U_conv1',
                                       reuse=self.reuse ,
                                       conv_init=conv_init1,
                                       bias_init=bias_init1,
                                       beta_init=beta_init1,
                                       gamma_init=gamma_init1,
                                       moving_mean_init=moving_mean_init1,
                                       moving_var=moving_var1)

            conv2 = self.layers.init_conv3d(conv1,
                                       filters=filters2,
                                       kernel_size=kernel_size,
                                       padding=padding2,
                                       dilation_rate=2,
                                       is_training=is_training,
                                       trainable=self.trainable,
                                       scope=scope + 'U_conv2',
                                       reuse=self.reuse,
                                       conv_init=conv_init2,
                                       bias_init=bias_init2,
                                       beta_init=beta_init2,
                                       gamma_init=gamma_init2,
                                       moving_mean_init=moving_mean_init2,
                                       moving_var=moving_var2)

            crop = conv2[:,
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2),
                   tf.to_int32(in_size / 2) - tf.to_int32(crop_size / 2) - 1:
                   tf.to_int32(in_size / 2) + tf.to_int32(crop_size / 2), :]

            return conv2, crop



    # ==================================================
    def unet(self, img_row1, input_dim, is_training, reuse=False):
        self.reuse=reuse
        in_size0 = tf.to_int32(0)
        in_size1 = tf.to_int32(input_dim)
        in_size2 = tf.to_int32(in_size1)  # conv stack
        in_size3 = tf.to_int32((in_size2))  # level_design1
        in_size4 = tf.to_int32(in_size3 / 2)  # downsampleing1+level_design2
        in_size5 = tf.to_int32(in_size4 / 2 - 4)  # downsampleing2+level_design3
        crop_size0 = tf.to_int32(0)
        crop_size1 = tf.to_int32(2 * in_size5 + 1)
        crop_size2 = tf.to_int32(2 * (crop_size1 - 4) + 1)

        # == == == == == == == == == == == == == == == == == == == == == ==
        # level 1 of unet

        [self.level_ds1, self.crop1] = self.level_design(img_row1, filters1=16, filters2=32,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size3,
                                               crop_size=crop_size2,
                                               padding1='same',
                                               padding2='same',
                                               scope='U_LD_DS1',
                                               conv_init1=self.conv_init1_ld1,
                                               bias_init1=self.bias_init1_ld1,
                                               beta_init1=self.beta_init1_ld1,
                                               gamma_init1=self.gamma_init1_ld1,
                                               moving_mean_init1=self.moving_mean_init1_ld1,
                                               moving_var1=self.moving_var1_ld1,
                                               conv_init2=self.conv_init2_ld1,
                                               bias_init2=self.bias_init2_ld1,
                                               beta_init2=self.beta_init2_ld1,
                                               gamma_init2=self.gamma_init2_ld1,
                                               moving_mean_init2=self.moving_mean_init2_ld1,
                                               moving_var2=self.moving_var2_ld1)
        with tf.variable_scope('U_maxpool1'):
            self.pool1 = tf.layers.max_pooling3d(inputs=self.level_ds1, pool_size=(2, 2, 2), strides=(2, 2, 2))
        # level 2 of unet
        [self.level_ds2, self.crop2] = self.level_design(self.pool1, filters1=32, filters2=64,
                                               is_training=is_training,
                                               kernel_size=3,
                                               in_size=in_size4,
                                               crop_size=crop_size1,
                                               padding1='same',
                                               padding2='same',
                                               scope='U_LD_DS2',
                                                 conv_init1=self.conv_init1_ld2,
                                                 bias_init1=self.bias_init1_ld2,
                                                 beta_init1=self.beta_init1_ld2,
                                                 gamma_init1=self.gamma_init1_ld2,
                                                 moving_mean_init1=self.moving_mean_init1_ld2,
                                                 moving_var1=self.moving_var1_ld2,
                                                 conv_init2=self.conv_init2_ld2,
                                                 bias_init2=self.bias_init2_ld2,
                                                 beta_init2=self.beta_init2_ld2,
                                                 gamma_init2=self.gamma_init2_ld2,
                                                 moving_mean_init2=self.moving_mean_init2_ld2,
                                                 moving_var2=self.moving_var2_ld2)
        with tf.variable_scope('U_maxpool2'):
            self.pool2 = tf.layers.max_pooling3d(inputs=self.level_ds2, pool_size=(2, 2, 2), strides=(2, 2, 2))

        # level 3 of unet
        [self.level_us1, self.crop0] = self.level_design(self.pool2, filters1=64, filters2=128,
                                               is_training=is_training, kernel_size=3, in_size=in_size0,
                                               crop_size=crop_size0,
                                               padding1='same',
                                               padding2='valid',
                                               scope='U_LD_US1',
                                                 conv_init1=self.conv_init1_ld3,
                                                 bias_init1=self.bias_init1_ld3,
                                                 beta_init1=self.beta_init1_ld3,
                                                 gamma_init1=self.gamma_init1_ld3,
                                                 moving_mean_init1=self.moving_mean_init1_ld3,
                                                 moving_var1=self.moving_var1_ld3,
                                                 conv_init2=self.conv_init2_ld3,
                                                 bias_init2=self.bias_init2_ld3,
                                                 beta_init2=self.beta_init2_ld3,
                                                 gamma_init2=self.gamma_init2_ld3,
                                                 moving_mean_init2=self.moving_mean_init2_ld3,
                                                 moving_var2=self.moving_var2_ld3)
        # with tf.variable_scope('U_upsample1'):
        #     deconv1 = self.upsampling3d.upsampling3d(level_us1,
        #                                              'U_3dUS1',
        #                                              trainable=self.trainable)
        #     #
        #     conc12 = tf.concat([crop2, deconv1], 4)
        #
        # # level 2 of unet
        # [level_us2, crop0] = self.level_design(conc12, filters1=128, filters2=64,
        #                                        is_training=is_training, kernel_size=3, in_size=in_size0,
        #                                        crop_size=crop_size0,
        #                                        padding1='same',
        #                                        padding2='valid',
        #                                        scope='U_LD_US2')
        # with tf.variable_scope('U_upsample2'):
        #     deconv2 = self.upsampling3d.upsampling3d(level_us2,
        #                                              'U_3dUS2',
        #                                              trainable=self.trainable)
        #     # with tf.variable_scope('U_concat23'):
        #     conc23 = tf.concat([crop1, deconv2], 4)
        #
        # # level 1 of unet
        # [level_us3, crop0] = self.level_design(conc23,
        #                                        filters1=64,
        #                                        filters2=32,
        #                                        is_training=is_training,
        #                                        kernel_size=1,
        #                                        in_size=in_size0,
        #                                        crop_size=crop_size0,
        #                                        padding1='same',
        #                                        padding2='same',
        #                                        scope='U_LD_US3'
        #                                        )
        # with tf.variable_scope('U_conv'):
        #     conv1 = self.layers.conv3d(level_us3,
        #                                filters=16,
        #                                kernel_size=3,
        #                                padding='same',
        #                                dilation_rate=1,
        #                                is_training=is_training,
        #                                trainable=self.trainable,
        #                                scope='U_conv1',
        #                                reuse=self.reuse)
        #
        # # == == == == == == == == == == == == == == == == == == == == == ==
        #
        #
        # # classification layer:
        # with tf.variable_scope('U_y'):
        #     y = self.layers.conv3d(conv1,
        #                            filters=1,
        #                            kernel_size=1,
        #                            padding='same',
        #                            dilation_rate=1,
        #                            is_training=is_training,
        #                            trainable=self.trainable,
        #                            scope='U_y',
        #                            reuse=self.reuse)
        #
        #
        #     # tf.self.layers.conv3d(conv1, filters=self.class_no,
        #     #                  kernel_size=1,
        #     #                  padding='same',
        #     #                  strides=(1, 1, 1),
        #     #                  activation=None,
        #     #                  dilation_rate=(1, 1,1),
        #     #                  name='last_conv'+self.log_ext)
        #
        # print(' total number of variables %s' % (
        #     np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
        #
        return  self.level_us1  # ,dense_out1,dense_out2,dense_out3,dense_out5,dense_out6
        #
