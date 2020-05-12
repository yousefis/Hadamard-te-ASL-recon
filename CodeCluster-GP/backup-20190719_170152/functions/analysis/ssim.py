from skimage.measure import compare_ssim as skissim

def ssim(noisy_img, gt,min_,max_):
    '''

    :param noisy_img:
    :param gt:
    :param min_:
    :param max_:
    :return:
    '''
    ssim_ = skissim(gt,noisy_img, data_range=max_ - min_)
    return ssim_
