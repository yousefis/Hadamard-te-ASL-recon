import tensorflow as tf
from niftynet.layer.layer_util import expand_spatial_params, infer_spatial_rank

def gaussian_1d(sigma=1.5, truncated=3.0):
    if sigma <= 0:
        return tf.constant(0.0)

    tail = int(sigma * truncated + 0.5)
    sigma_square = sigma **2
    k = [(-0.5 * x * x) / sigma_square for x in range(-tail, tail + 1)]
    k = tf.nn.softmax(k)
    return k

def do_conv(input_tensor, dim):
    spatial_rank = infer_spatial_rank(input_tensor)

    assert dim < spatial_rank
    if dim < 0:
        return input_tensor

    _sigmas = expand_spatial_params(input_param=1.5, spatial_rank=spatial_rank, param_type=float)
    _truncate = expand_spatial_params(input_param=3.0, spatial_rank=spatial_rank, param_type=float)

    # squeeze the kernel to be along the 'dim'
    new_kernel_shape = [1] * (spatial_rank + 2)
    new_kernel_shape[dim] = -1
    kernel_tensor = gaussian_1d(sigma=_sigmas[dim], truncated=_truncate[dim])
    kernel_tensor = tf.reshape(kernel_tensor, new_kernel_shape)

    # split channels and do smoothing respectively
    chn_wise_list = tf.unstack(do_conv(input_tensor, dim - 1), axis=-1)
    output_tensor = [tf.nn.convolution(input=tf.expand_dims(chn, axis=-1), filter=kernel_tensor, padding='VALID',
                                       strides=[1] * spatial_rank) for chn in chn_wise_list]
    return tf.concat(output_tensor, axis=-1)

def SSIM(x1, x2, max_val=1.0,axes=None):

    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2
    spatial_rank = infer_spatial_rank(x1)
    frameReference = tf.cast(x1, tf.float32)
    frameUnderTest = tf.cast(x2, tf.float32)
    frameReference_square = tf.square(frameReference)
    frameUnderTest_square = tf.square(frameUnderTest)
    frameReference_frameUnderTest = frameReference * frameUnderTest

    mu1 = do_conv(frameReference, spatial_rank-1)
    mu2 = do_conv(frameUnderTest, spatial_rank-1)
    mu1_square = tf.square(mu1)
    mu2_square = tf.square(mu2)
    mu1_mu2 = mu1 * mu2
    sigma1_square = do_conv(frameReference_square, spatial_rank-1)
    sigma1_square = sigma1_square - mu1_square
    sigma2_square = do_conv(frameUnderTest_square, spatial_rank-1)
    sigma2_square = sigma2_square - mu2_square
    sigma12 = do_conv(frameReference_frameUnderTest, spatial_rank-1)
    sigma12 = sigma12 - mu1_mu2

    numerator = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    denominator = ((mu1_square + mu2_square + C1) * (sigma1_square + sigma2_square + C2))
    ssim_map = numerator / denominator

    if spatial_rank == 3 and axes==None:
        axes = tf.constant([-4, -3, -2, -1, 0], dtype=tf.int32)
    elif not spatial_rank == 3 and axes==None:
        axes = tf.constant([-3, -2, -1, 0], dtype=tf.int32)

    mssim = tf.reduce_mean(ssim_map, axis=axes)

    return mssim,ssim_map

def GPU_SSIM(x1, x2, max_val):

    C1 = (0.01 * max_val)**2
    C2 = (0.03 * max_val)**2
    spatial_rank = infer_spatial_rank(x1)
    frameReference = tf.cast(x1, tf.float32)
    frameUnderTest = tf.cast(x2, tf.float32)
    frameReference_square = tf.square(frameReference)
    frameUnderTest_square = tf.square(frameUnderTest)
    frameReference_frameUnderTest = frameReference * frameUnderTest

    mu1 = do_conv(frameReference, spatial_rank-1)
    mu2 = do_conv(frameUnderTest, spatial_rank-1)
    mu1_square = tf.square(mu1)
    mu2_square = tf.square(mu2)
    mu1_mu2 = mu1 * mu2
    sigma1_square = do_conv(frameReference_square, spatial_rank-1)
    sigma1_square = sigma1_square - mu1_square
    sigma2_square = do_conv(frameUnderTest_square, spatial_rank-1)
    sigma2_square = sigma2_square - mu2_square
    sigma12 = do_conv(frameReference_frameUnderTest, spatial_rank-1)
    sigma12 = sigma12 - mu1_mu2

    numerator = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    denominator = ((mu1_square + mu2_square + C1) * (sigma1_square + sigma2_square + C2))
    ssim_map = numerator / denominator

    if spatial_rank == 3:
        axes = tf.constant([-4, -3, -2, -1, 0], dtype=tf.int32)
    else:
        axes = tf.constant([-3, -2, -1, 0], dtype=tf.int32)

    mssim = tf.reduce_mean(1-ssim_map, axis=axes)

    return mssim,ssim_map

def SSIM_loss(x1, x2, max_val=1.0):
    ssim, _ = SSIM(x1, x2, max_val=max_val)
    return 1.0 - ssim