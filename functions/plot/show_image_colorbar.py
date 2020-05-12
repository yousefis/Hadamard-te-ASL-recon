import functions.analysis as anly
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':

    # parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
    parent_path = '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'

    mse_path = 'Workshop_log/MSE/experiment-2/results/'
    ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
    perceptual_path = 'Log_perceptual/regularization/perceptual-0/results/'
    multistage_path = 'multi_stage/experiment-2/results/'
    subject = '20_PP4_50_cinema1/'

    from_ = 20
    to_ = 85
    slice = 49
    max = 30  # np.max(img_arry)
    min = -1  # np.min(img_arry)
    vmin=-max
    fig = plt.figure()
    # fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(9, 6),
    #                         subplot_kw={'xticks': [], 'yticks': []})
    type = 'angi_'
    # ============================
    flag = True
    # ============================
    plt.rcParams.update({'font.size': 16})

    fontdicty = {'fontsize': 16,
                 'weight': 'bold',
                 'verticalalignment': 'baseline',
                 'horizontalalignment': 'center'}



    fig, axs = plt.subplots(5, 14,  constrained_layout=True)
    error_rng= 1
    plds=[265,565,865,1265,1665,2065,2665]
    for i in range(0, 7):
        scan = type + str(i) + '.mha'
        mse_img_path = parent_path + mse_path + subject + scan
        ssim_img_path = parent_path + ssim_path + subject + scan
        perceptual_img_path = parent_path + perceptual_path + subject + scan
        multistage_img_path = parent_path + multistage_path + subject + scan
        gt_path = parent_path + ssim_path + subject + 'GT/angio_' + str(i) + '.mha'

        img_mse = sitk.GetArrayFromImage(sitk.ReadImage(mse_img_path))
        img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))
        img_perceptual = sitk.GetArrayFromImage(sitk.ReadImage(perceptual_img_path))
        img_multistage = sitk.GetArrayFromImage(sitk.ReadImage(multistage_img_path))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))

        # max_angio = 2.99771428108
        # print(anly.analysis(img_mse,
        #               img_gt, 0, max_angio))
        # print(anly.analysis(img_ssim,
        #                     img_gt, 0, max_angio))
        # print(anly.analysis(img_perceptual,
        #                     img_gt, 0, max_angio))
        # print(anly.analysis(img_multistage,
        #                     img_gt, 0, max_angio))
        # =================================
        im = axs[0, i].imshow(((img_gt[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[0, i])
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('Target', rotation=0, fontsize=20, labelpad=45)
        im2 = axs[0, 7 * 1 + i].imshow(((img_gt[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])),
                                       vmin=-error_rng, vmax=error_rng, cmap='jet')
        plt.sca(axs[0, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        # =================================
        im = axs[1, i].imshow(((img_mse[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[1, i])
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('MSE', rotation=0, fontsize=20, labelpad=45)
        im2 = axs[1, 7 * 1 + i].imshow(((img_mse[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])),
                                       vmin=-error_rng, vmax=error_rng, cmap='jet')
        plt.sca(axs[1, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])

        # =================================
        im = axs[2, i].imshow(((img_ssim[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[2, i])
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('SSIM', rotation=0, fontsize=20, labelpad=45)
        im2 = axs[2, 7 * 1 + i].imshow(((img_ssim[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])),
                                       vmin=-error_rng, vmax=error_rng, cmap='jet')
        plt.sca(axs[2, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        #=================================
        im = axs[3, i].imshow(((img_perceptual[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[3, i])
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('PL', rotation=0, fontsize=20, labelpad=45)
        im2 = axs[3, 7 * 1 + i].imshow(
            ((img_perceptual[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])), vmin=-error_rng, vmax=error_rng,
            cmap='jet')
        plt.sca(axs[3, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        # =================================
        im = axs[4, i].imshow(((img_multistage[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[4, i])
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.ylabel('ML-SSIM', rotation=0, fontsize=20, labelpad=45)
        im2 = axs[4, 7 * 1 + i].imshow(
            ((img_multistage[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])), vmin=-error_rng, vmax=error_rng,
            cmap='jet')
        plt.sca(axs[4, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])

    # =================================
    fig.colorbar(im, ax=axs[4, 0:7], shrink=0.6, extend="both", location='bottom')
    fig.colorbar(im2, ax=axs[4, 6:15], shrink=0.6, extend="both", location='bottom')

    path = parent_path+'fig_miccai2019/'
    fig_nm = path + type + 'slice' + str(slice)
    plt.savefig(fig_nm + '.eps', format='eps', dpi=1000)

    print(fig_nm)
    # fig.colorbar(im, ax=axs[3, 0:7], shrink=1.0,pad=0.0,extend="both", location='bottom', fraction=.1)
    # fig.colorbar(im2, ax=axs[3, 7:14], shrink=1.0, pad=0.0,extend="both",location='bottom', fraction=.1)
    plt.subplots_adjust(wspace=.0, hspace=.0)
    plt.show()