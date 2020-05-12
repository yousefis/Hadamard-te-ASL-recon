import numpy as np

def snr(noisy,gt):
    '''

    :param noisy:
    :param gt:
    :return:
    '''
    height=103 #size of the images is 103**3

    numerator = np.sqrt(np.sum(np.sum((noisy) ** 2)) / height ** 3)
    denominator = np.sqrt(np.sum(np.sum((noisy - gt) ** 2)) / height ** 3)
    return numerator/denominator

def signaltonoise(a, axis, ddof):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis = axis, ddof = ddof)
    return np.where(sd == 0, 0, m / sd)