import numpy as np
import math
def psnr(img1, img2,PIXEL_MAX):
    '''

    :param img1:
    :param img2:
    :param PIXEL_MAX:
    :return:
    '''
    eps = 10e-6
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 0
    if PIXEL_MAX<=0:
        return 0
    if math.log10(PIXEL_MAX / (math.sqrt(mse)+eps))<0:
        return 0
    return 20 * math.log10(PIXEL_MAX / (math.sqrt(mse)+eps))
