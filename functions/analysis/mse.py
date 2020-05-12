
def mse(noisy_img, gt):
    '''

    :param noisy_img:
    :param gt:
    :return:
    '''
    mse = ((noisy_img - gt)**2).mean(axis=0).mean(axis=0).mean(axis=0)
    return mse
