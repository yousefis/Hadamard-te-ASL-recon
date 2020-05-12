from .snr import *
from .ssim import *
from .mse import *
from .psnr import *
def analysis(result,gt, min, max):
    '''
    :param result:
    :param gt:
    :param min:
    :param max:
    :return:
    '''

    PSNR = psnr(result, gt, PIXEL_MAX=max)
    SNR = snr(result, gt)
    MSE = mse(result, gt)
    SSIM = ssim(result, gt, min_=min, max_=max)
    analysis_dic = {'PSNR':PSNR, 'SNR':SNR,'MSE':MSE,'SSIM':SSIM}
    return analysis_dic