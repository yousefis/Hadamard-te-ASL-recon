import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import functions.layers.conv_kernel as convKernel
class downsampler:
    def __init__(self):
        a=1
    def array_to_sitk(self, array_input, origin=None, spacing=None, direction=None, is_vector=False, im_ref=None):
        if origin is None:
            origin = [0, 0, 0]
        if spacing is None:
            spacing = [1, 1, 1]
        if direction is None:
            direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        sitk_output = sitk.GetImageFromArray(array_input, isVector=is_vector)
        if im_ref is None:
            sitk_output.SetOrigin(origin)
            sitk_output.SetSpacing(spacing)
            sitk_output.SetDirection(direction)
        else:
            sitk_output.SetOrigin(im_ref.GetOrigin())
            sitk_output.SetSpacing(im_ref.GetSpacing())
            sitk_output.SetDirection(im_ref.GetDirection())
        return sitk_output
    def downsampler(self,input, down_scale, kernel_name='bspline', normalize_kernel=True, a=-0.5, default_pixel_value=0):
        """
        Downsampling tensor
        :param input: can be a 2D or 3D numpy array or sitk image
        :param down_scale: an integer value!
        :param kernel_name:
        :param normalize_kernel:
        :param a:
        :param default_pixel_value:
        :return: output: can be a numpy array or sitk image based on the input
        """
        kernelDimension =3# len(np.shape(input_numpy))
        # input_numpy = np.expand_dims(input_numpy[np.newaxis], axis=-1)
        if down_scale == 2:
            kernel_size = 7
            padSize = (np.floor(kernel_size / 2) - 1).astype(np.int)
        elif down_scale == 4:
            kernel_size = 15
            padSize = (np.floor(kernel_size / 2) - 2).astype(np.int)
        else:
            raise ValueError('kernel_size is not defined for down_scale={}'.format(str(down_scale)))

        kenelStrides = tuple([down_scale] * kernelDimension)

        # x = tf.placeholder(tf.float32, shape=np.shape(input), name="InputImage")
        x_pad = tf.pad(input, ([0, 0], [padSize, padSize], [padSize, padSize], [padSize, padSize], [0, 0]), constant_values=default_pixel_value)
        convKernelGPU = convKernel.convDownsampleKernel(kernel_name, kernelDimension, kernel_size, normalizeKernel=normalize_kernel, a=a)

        convKernelGPU = np.expand_dims(convKernelGPU, -1)
        convKernelGPU = np.expand_dims(convKernelGPU, -1)
        convKernelGPU = tf.constant(convKernelGPU)


        ds = tf.concat([tf.nn.convolution(x_pad[:, :, :, :, i, tf.newaxis], convKernelGPU, 'VALID', strides=kenelStrides)
                        for i in range(int(x_pad.get_shape()[4]))], axis=-1)

        return ds

    def downsampler_gpu(self,input, down_scale, kernel_name='bspline', normalize_kernel=True, a=-0.5, default_pixel_value=0):
        """
        Downsampling numpy by gpu
        :param input: can be a 2D or 3D numpy array or sitk image
        :param down_scale: an integer value!
        :param kernel_name:
        :param normalize_kernel:
        :param a:
        :param default_pixel_value:
        :return: output: can be a numpy array or sitk image based on the input
        """

        if isinstance(input, sitk.Image):
            input_numpy = sitk.GetArrayFromImage(input)
            mode = 'sitk'
        else:
            input_numpy = input
            mode = 'numpy'
        if not isinstance(down_scale, int):
            'type is:'
            print(type(down_scale))
            raise ValueError('down_scale should be integer. now it is {} with type of '.format(down_scale)+type(down_scale))

        kernelDimension =3# len(np.shape(input_numpy))
        # input_numpy = np.expand_dims(input_numpy[np.newaxis], axis=-1)
        if down_scale == 2:
            kernel_size = 7
            padSize = (np.floor(kernel_size / 2) - 1).astype(np.int)
        elif down_scale == 4:
            kernel_size = 15
            padSize = (np.floor(kernel_size / 2) - 2).astype(np.int)
        else:
            raise ValueError('kernel_size is not defined for down_scale={}'.format(str(down_scale)))

        kenelStrides = tuple([down_scale] * kernelDimension)

        tf.reset_default_graph()
        sess = tf.Session()
        x = tf.placeholder(tf.float32, shape=np.shape(input_numpy), name="InputImage")
        x_pad = tf.pad(x, ([0, 0], [padSize, padSize], [padSize, padSize], [padSize, padSize], [0, 0]), constant_values=default_pixel_value)
        convKernelGPU = convKernel.convDownsampleKernel(kernel_name, kernelDimension, kernel_size, normalizeKernel=normalize_kernel, a=a)
        convKernelGPU = np.expand_dims(convKernelGPU, -1)
        convKernelGPU = np.expand_dims(convKernelGPU, -1)

        convKernelGPU = tf.constant(convKernelGPU)
        y = tf.nn.convolution(x_pad, convKernelGPU, 'VALID', strides=kenelStrides)
        sess.run(tf.global_variables_initializer())
        [output_numpy] = sess.run([y], feed_dict={x: input_numpy})
        # if kernelDimension == 2:
        #     output_numpy = output_numpy[0, :, :, 0]
        # if kernelDimension == 3:
        #     output_numpy = output_numpy[0, :, :, :, 0]
        #
        # if mode == 'numpy':
        #     output = output_numpy
        # elif mode == 'sitk':
        #     output = self.array_to_sitk(output_numpy, origin=input.GetOrigin(),
        #                            spacing=tuple(i * down_scale for i in input.GetSpacing()), direction=input.GetDirection())
        return output_numpy
