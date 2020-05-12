import numpy as np

def snr(noisy,gt):
    '''

    :param noisy:
    :param gt:
    :return:
    '''
    numerator = np.sum(np.sum((noisy) ** 2))
    denominator = np.sum(np.sum((noisy - gt) ** 2))
    return numerator/denominator

