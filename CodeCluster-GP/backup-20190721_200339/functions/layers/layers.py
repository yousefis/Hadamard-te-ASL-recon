import numpy as np
import tensorflow as tf
class layers:
    def __init__(self):
        a=1

    def conv3d(self, input, filters, kernel_size, padding, dilation_rate, is_training, trainable, scope,reuse):
        with tf.variable_scope(scope):
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
            bn = tf.layers.batch_normalization(conv,
                                               training=is_training,
                                               renorm=False,
                                               trainable=trainable,
                                               name= scope+'_bn',
                                               reuse=reuse)
            bn = tf.nn.relu(bn)
        return bn

    def conv3d_nonscope(self, input, filters, kernel_size, padding, dilation_rate, is_training, trainable, scope, reuse,activation):
        conv = tf.layers.conv3d(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation=None,
                                dilation_rate=dilation_rate,
                                trainable=trainable,
                                name=scope + '_conv3d',
                                reuse=reuse
                                )
        bn = tf.layers.batch_normalization(conv,
                                           training=is_training,
                                           renorm=False,
                                           trainable=trainable,
                                           name=scope + '_bn',
                                           reuse=reuse)
        if activation=='ReLU':
            bn = tf.nn.relu(bn)
        elif activation=='ReakyReLU':
            bn = tf.nn.leaky_relu(bn)
        return bn
        # ==================================================

    def bilinear_up_kernel(self, dim=3, kernel_size=3):
        center = kernel_size // 2
        if dim == 3:
            indices = [None] * dim
            indices[0], indices[1], indices[2] = np.meshgrid(np.arange(0, 3), np.arange(0, 3), np.arange(0, 3),
                                                             indexing='ij')
            for i in range(0, dim):
                indices[i] = indices[i] - center
            distance_to_center = np.absolute(indices[0]) + np.absolute(indices[1]) + np.absolute(indices[2])
            kernel = (np.ones(np.shape(indices[0])) / (np.power(2, distance_to_center))).astype(np.float32)

        return kernel

    # ========================
    def init_conv3d(self, input, filters, kernel_size, padding, dilation_rate, is_training, trainable, scope,reuse,
               conv_init,bias_init,beta_init,gamma_init,moving_mean_init,moving_var):
        conv = tf.layers.conv3d(input,
                                filters=filters,
                                kernel_size=kernel_size,
                                padding=padding,
                                activation=None,
                                dilation_rate=dilation_rate,
                                trainable=trainable,
                                name=scope + '_conv3d',
                                reuse=reuse,
                                kernel_initializer=tf.constant_initializer(conv_init[0]),
                                bias_initializer=tf.constant_initializer(bias_init[0])
                                )
        bn = tf.layers.batch_normalization(conv,
                                           training=is_training,
                                           renorm=False,
                                           trainable=trainable,
                                           name=scope + '_bn',
                                           reuse=reuse,
                                           beta_initializer=tf.constant_initializer(beta_init[0]),
                                           gamma_initializer=tf.constant_initializer(gamma_init[0]),
                                           moving_mean_initializer=tf.constant_initializer(moving_mean_init[0]),
                                           moving_variance_initializer=tf.constant_initializer(moving_var[0]),
                                           )
        bn = tf.nn.relu(bn)
        return bn
    def conv3d_transpose(self, input_layer, filters, kernel_size, padding='valid', bn_training=None,
                         strides=(2, 2, 2),
                         scope=None, activation=None, use_bias=False, initializer=None, trainable=True,reuse=False):
        """

        :param input_layer:
        :param filters:
        :param kernel_size:
        :param padding:
        :param bn_training: None: no batch_normalization, 1: batch normalization in training mode, 0: batch _normalization in test mode:
        :param strides:
        :param scope:
        :param activation:
        :param use_bias:
        :param initializer: None (default) or 'trilinear'
        :param trainable:
        :return:
        """

        kernel_initializer = None
        if initializer is not None:
            if initializer == 'trilinear':
                conv_kernel_trilinear = self.bilinear_up_kernel(dim=3)
                conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
                conv_kernel_trilinear = np.repeat(conv_kernel_trilinear, filters, axis=-1)
                conv_kernel_trilinear = np.expand_dims(conv_kernel_trilinear, -1)
                conv_kernel_trilinear = np.repeat(conv_kernel_trilinear, int(input_layer.get_shape()[4]), axis=-1)
                # size of the conv_kernel should be [3, 3, 3, input_layer.get_shape()[4], filters]: double checked.
                kernel_initializer = tf.constant_initializer(conv_kernel_trilinear)
            else:
                raise ValueError(
                    'initializer=' + initializer + ' is not defined in conv3d_transpose. Valid options: "trilinear"')

        with tf.variable_scope(scope):
            net = tf.layers.conv3d_transpose(input_layer, filters, kernel_size,
                                             padding=padding,
                                             strides=strides,
                                             kernel_initializer=kernel_initializer,
                                             use_bias=use_bias,
                                             trainable=trainable,
                                             reuse=reuse)
            if bn_training is not None:
                net = tf.layers.batch_normalization(net, training=bn_training)
            if activation is not None:
                if activation == 'LReLu':
                    net = tf.nn.leaky_relu(net)
                elif activation == 'ReLu':
                    net = tf.nn.relu(net)
                elif activation == 'ELu':
                    net = tf.nn.elu(net)
                else:
                    raise ValueError(
                        'activation=' + activation + ' is not defined in tfu.conv3d. Valid options: "ReLu", "LReLu"')
        return net