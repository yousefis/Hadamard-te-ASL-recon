import tensorflow as tf
import numpy as np
from functions.layers.kernels import bilinear_up_kernel
class upsampling:
    def __init__(self):
        a=1
        # ==================================================


    def upsampling3d(self,input_layer, scope, scale=2, interpolator='trilinear', padding_mode='SYMMETRIC',
                     padding_constant=0, trainable=False, padding='valid', output_shape_3d=None):
        """

        Key features:
            - It can have any upsampling kernel. (nearest neighbor and trilinear is implemented)
            - It has the padding_mode 'CONSTANT', 'REFLECT', 'SYMMETRIC'

        Limitation
            - It is limited to integer value of strides.
            - 'valid' mode is implemented, to_do: 'same'


        interpolator='nearest_neighbor':
            we use tf.keras.layers.UpSampling3D

        interpolator='trilinear':
            we use tf.nn.conv3d_transpose separately for each feature map.

        :param input_layer:
        :param scope:
        :param scale:
        :param interpolator: 'trilinear', 'nearest_neighbor'
        :param padding_mode : 'CONSTANT', 'REFLECT', 'SYMMETRIC'
        :param padding_constant
        :param trainable
        :param padding onle 'valid' mode is implemented
        :param output_shape: it can be defined in the trilinear mode. if not the default is scale*input_layer.size()+1

        :return:
        """
        if padding.upper() != 'VALID':
            print('upsampling3d is only implemented for "VALID" mode, TODO: "SAME"')

        pad_size = 1  # the input will be padded in order to prevent border effect.
        if interpolator == 'nearest_neighbor':
            with tf.variable_scope(scope):
                upsample_layer = tf.keras.layers.UpSampling3D(size=(2, 2, 2), data_format='channels_last',
                                                              trainable=trainable)
                net = upsample_layer.__call__(tf.pad(input_layer,
                                                     ([0, 0], [pad_size, pad_size], [pad_size, pad_size],
                                                      [pad_size, pad_size], [0, 0]),
                                                     mode=padding_mode,
                                                     constant_values=padding_constant))

            return net[:, 2 * pad_size:-2 * pad_size + 1, 2 * pad_size:-2 * pad_size + 1,
                   2 * pad_size:-2 * pad_size + 1, :]

        if interpolator == 'trilinear':
            with tf.variable_scope(scope) as Scope:
                conv_kernel_trilinear = bilinear_up_kernel(dim=3)
                conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
                conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
                # size of the conv_kernel should be [3, 3, 3, input_layer.get_shape()[4], filters]: double checked.
                kernel_initializer = tf.constant_initializer(conv_kernel_trilinear)

                # output_shape = input_layer[:, :, :, :, 0, tf.newaxis].get_shape().as_list()
                output_shape=tf.shape(input_layer)
                if output_shape_3d is None:
                    output_shape1 = scale * (output_shape[1] + 2 * pad_size) + 1
                    output_shape2 = scale * (output_shape[2] + 2 * pad_size) + 1
                    output_shape3 = scale * (output_shape[3] + 2 * pad_size) + 1
                else:
                    output_shape1 = output_shape_3d[0] + 4 * pad_size
                    output_shape2 = output_shape_3d[1] + 4 * pad_size
                    output_shape3 = output_shape_3d[2] + 4 * pad_size

                # output_shape_tf = tf.stack(
                #     [tf.shape(input_layer)[0],  output_shape[1], output_shape[2], output_shape[3], output_shape[4]])
                output_shape_tf = tf.stack(
                    [tf.shape(input_layer)[0], output_shape1, output_shape2, output_shape3, 1])
                try:
                    filter_transposed = tf.get_variable("kernel_transposed_3d",
                                                    shape=(3, 3, 3, 1, 1),
                                                    dtype=tf.float32,
                                                    initializer=kernel_initializer,
                                                    trainable=trainable)
                except ValueError:
                    Scope.reuse_variables()
                    filter_transposed = tf.get_variable("kernel_transposed_3d",
                                                    shape=(3, 3, 3, 1, 1),
                                                    dtype=tf.float32,
                                                    initializer=kernel_initializer,
                                                    trainable=trainable)

                net =tf.concat([tf.nn.conv3d_transpose(tf.pad(input_layer,
                                                           ([0, 0],
                                                            [pad_size, pad_size],
                                                            [pad_size, pad_size],
                                                            [pad_size, pad_size],
                                                            [0, 0]),
                                                           mode=padding_mode,
                                                           constant_values=padding_constant)[:, :, :, :, i, tf.newaxis],
                                                    filter=filter_transposed,
                                                    strides=(1, scale, scale, scale, 1),
                                                    padding=padding.upper(),
                                                    output_shape=output_shape_tf)
                            for i in range(int(input_layer.get_shape()[4]))], axis=-1)


            return net[:, 2 * pad_size:-2 * pad_size, 2 * pad_size:-2 * pad_size, 2 * pad_size:-2 * pad_size, :]

            # ==================================================

