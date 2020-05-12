import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
    ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
    perceptual_path = 'Log_perceptual/regularization/perceptual-0/results/'
    multistage_path = 'multi_stage/experiment-2/results/'
    subject = '18_PP5_53_cinema2/'
    type = 'angi_'
    fig = plt.figure()
    min_=-1
    max_=700
    for i in range(7):
        scan = type + str(i) + '.mha'
        ssim_img_path = parent_path + ssim_path + subject + scan
        perceptual_img_path = parent_path + perceptual_path + subject + scan
        multistage_img_path = parent_path + multistage_path + subject + scan
        gt_path = parent_path + ssim_path + subject + 'GT/' + 'angio_' + str(i) + '.mha'

        img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))
        img_perceptual = sitk.GetArrayFromImage(sitk.ReadImage(perceptual_img_path))
        img_multistage = sitk.GetArrayFromImage(sitk.ReadImage(multistage_img_path))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        axis=1
        ssim_mra=np.sum(img_ssim,axis=axis)
        perceptual_mra=np.sum(img_perceptual,axis=axis)
        multistage_mra=np.sum(img_multistage,axis=axis)
        gt_mra=np.sum(img_gt,axis=axis)
        cmap='jet'
        ax1 = fig.add_subplot(4,7,i+1)
        im2 = ax1.imshow(np.rot90(np.rot90(ssim_mra)), vmin=min_, vmax=max_, cmap=cmap)

        ax1 = fig.add_subplot(4, 7, 7+i+1)
        im2 = ax1.imshow(np.rot90(np.rot90(perceptual_mra)), vmin=min_, vmax=max_, cmap=cmap)

        ax1 = fig.add_subplot(4, 7, 14 + i+1)
        im2 = ax1.imshow(np.rot90(np.rot90(multistage_mra)), vmin=min_, vmax=max_, cmap=cmap)

        ax1 = fig.add_subplot(4, 7, 21 + i+1)
        im2 = ax1.imshow(np.rot90(np.rot90(gt_mra)), vmin=min_, vmax=max_, cmap=cmap)

    plt.show()



