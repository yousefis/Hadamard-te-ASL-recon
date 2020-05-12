import numpy as np
import tensorflow as tf
def derivative_LoG( input):
    """
    derivative kernel for emphasizing the edges
    :param input:
    :return:
    """

    kernelDimension = 3  # len(np.shape(input_numpy))


    kenelStrides = tuple([1] * kernelDimension)

    x_pad = tf.cast(input,tf.float64)
    LoGKernel = [[[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]],
                         [[-1.,-1.,-1.],[-1.,26.,-1.],[-1.,-1.,-1.]],
                         [[-1.,-1.,-1.],[-1.,-1.,-1.],[-1.,-1.,-1.]]]
    LoGKernel = np.expand_dims(LoGKernel, -1)
    LoGKernel = np.expand_dims(LoGKernel, -1)
    LoGKernel = tf.constant(LoGKernel)

    GoL = tf.concat([tf.nn.convolution(x_pad[:, :, :, :, i, tf.newaxis], LoGKernel, 'VALID', strides=kenelStrides)
                    for i in range(int(x_pad.get_shape()[4]))], axis=-1)

    return GoL