import inspect
import os

import numpy as np
import tensorflow as tf

from functions.loss.perceptual_loss.vgg.pretrained_vgg.vgg16 import Vgg16
import tensorflow.contrib.slim as slim
from functions.loss.perceptual_loss.vgg.vgg_utils import *




class vgg_loader(Vgg16):
    # Input should be an rgb image [batch, height, width, 3]
    # values scaled [0, 1]

    def __init__(self,  data_dict,vgg_scope="vgg"):
        object.__init__(self)

        '''
        :param rgb:
        :param data_dict:
        :param train:
        '''
        # It's a shared weights data and used in various
        # member functions.
        self.vgg_scope=vgg_scope
        self.data_dict = data_dict
        with tf.variable_scope(None, vgg_scope):
            self.scope = tf.get_variable_scope()
        self.tensors_exist=False
        # start_time = time.time()

        # # rgb_scaled = rgb * 255.0
        # rgb_scaled = rgb
        # # Convert RGB to BGR
        # red, green, blue = tf.split(rgb_scaled, 3, 3)
        #
        # bgr = tf.concat([blue - VGG_MEAN[0],
        #                  green - VGG_MEAN[1],
        #                  red - VGG_MEAN[2]],
        #                 3)
    def vgg_feed(self,rgb):
        # with tf.variable_scope( self.vgg_scope,reuse=True):
        self.conv1_1 = self.conv_layer(rgb,  "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        # self.pool1= self.max_pool(self.conv1_2, "pool1")

        # self.conv2_1 = self.conv_layer(self.pool1, "conv2_1" )
        # self.conv2_2 = self.conv_layer(self.conv2_1,  "conv2_2" )
        # self.pool2 = self.max_pool(self.conv2_2, "pool2" )

        # self.conv3_1= self.conv_layer(self.pool2, "conv3_1" )
        # self.conv3_2 = self.conv_layer(self.conv3_1,  "conv3_2" )
        # self.conv3_3= self.conv_layer(self.conv3_2,  "conv3_3" )
        # self.pool3 = self.max_pool(self.conv3_3,  "pool3" )
        #
        # self.conv4_1 = self.conv_layer(self.pool3, "conv4_1" )
        # self.conv4_2  = self.conv_layer(self.conv4_1, "conv4_2" )
        # self.conv4_3  = self.conv_layer(self.conv4_2, "conv4_3" )
        # self.pool4 = self.max_pool(self.conv4_3,"pool4" )
        #
        # self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        # self.conv5_2 = self.conv_layer(self.conv5_1,  "conv5_2" )
        # self.conv5_3 = self.conv_layer(self.conv5_2,  "conv5_3")
        # self.pool5 = self.max_pool(self.conv5_3,  "pool5" )
        #
        # self.tensors_exist=True
    def debug(self):
        pass



