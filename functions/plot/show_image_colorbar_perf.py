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

    parent_path = '/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
    ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
    perceptual_path = 'Log_perceptual/regularization/perceptual-0/results/'
    multistage_path = 'multi_stage/experiment-2/results/'
    mse_path = 'Workshop_log/MSE/experiment-2/results/'
    subject = '20_PP2_18_cinema5/'

    from_ = 20
    to_ = 85
    slice = 49
    max = 6  # np.max(img_arry)
    min = -1  # np.min(img_arry)
    fig = plt.figure()
    # fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(9, 6),
    #                         subplot_kw={'xticks': [], 'yticks': []})
    type = 'perf_'
    # ============================
    flag = True
    # ============================
    plt.rcParams.update({'font.size': 16})

    # fontdicty = {'fontsize': 16,
    #              'weight': 'bold',
    #              'verticalalignment': 'baseline',
    #              'horizontalalignment': 'center'}



    fig, axs = plt.subplots(5, 14,  constrained_layout=True)

    scans_no=[6,5,4,3,2,1,0]
    for i in range(0, 7):
        scan = type + str(scans_no[i]) + '.mha'
        ssim_img_path = parent_path + ssim_path + subject + scan
        perceptual_img_path = parent_path + perceptual_path + subject + scan
        multistage_img_path = parent_path + multistage_path + subject + scan
        mse_img_path =parent_path + mse_path + subject + scan
        gt_path = parent_path + ssim_path + subject + 'GT/' + type + str(scans_no[i]) + '.mha'

        img_mse = sitk.GetArrayFromImage(sitk.ReadImage(mse_img_path))
        img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))
        img_perceptual = sitk.GetArrayFromImage(sitk.ReadImage(perceptual_img_path))
        img_multistage = sitk.GetArrayFromImage(sitk.ReadImage(multistage_img_path))
        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        # =========================
        im = axs[0, i].imshow(((img_mse[slice, from_:to_, from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[0, i])
        plt.xticks([])
        plt.yticks([])
        axs[0, i].set_aspect('equal')
        if i == 0:
            plt.ylabel('MSE')
        im2 = axs[0, 7 * 1 + i].imshow(((img_mse[slice, from_:to_, from_:to_] - img_gt[slice, from_:to_, from_:to_])),
                                       vmin=-max, vmax=max, cmap='jet')
        plt.sca(axs[0, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        axs[0, 7 * 1 + i].set_aspect('equal')
        #=========================

        im = axs[1,i].imshow(((img_ssim[slice,from_:to_,  from_:to_])), vmin=min, vmax=max, cmap='jet')
        plt.sca(axs[1,  i])
        plt.xticks([])
        plt.yticks([])
        axs[1,i].set_aspect('equal')
        if i == 0:
            plt.ylabel('SSIM')
        im2 = axs[1,7*1+i].imshow(((img_ssim[slice, from_:to_, from_:to_] - img_gt[ slice,from_:to_, from_:to_])),vmin=-max, vmax=max, cmap='jet')
        plt.sca(axs[1, 7*1+i])
        plt.xticks([])
        plt.yticks([])
        axs[1, 7 * 1 + i].set_aspect('equal')
        # =========================
        im = axs[2,i].imshow(((img_perceptual[slice,from_:to_,  from_:to_])), vmin=min, vmax=max,cmap='jet')
        plt.sca(axs[2,  i])
        plt.xticks([])
        plt.yticks([])
        axs[2,i].set_aspect('equal')
        if i == 0:
            plt.ylabel('PL')
        im2 = axs[2,7*1+i].imshow(((img_perceptual[slice, from_:to_, from_:to_]- img_gt[slice,from_:to_,  from_:to_])), vmin=-max, vmax=max,cmap='jet')
        plt.sca(axs[2, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        axs[2, 7 * 1 + i].set_aspect('equal')
        # =========================
        im = axs[3, i].imshow(((img_multistage[ slice,from_:to_, from_:to_])), vmin=min, vmax=max,cmap='jet')
        plt.sca(axs[3, i])
        plt.xticks([])
        plt.yticks([])
        axs[3, i].set_aspect('equal')
        if i == 0:
            plt.ylabel('ML-SSIM')
        im2 = axs[3, 7 * 1 + i].imshow(((img_multistage[slice,from_:to_,  from_:to_] - img_gt[slice,from_:to_,  from_:to_])),vmin=-max, vmax=max, cmap='jet')
        plt.sca(axs[3, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        axs[3, 7 * 1 + i].set_aspect('equal')
        # =========================

        im = axs[4, i].imshow(((img_gt[slice,from_:to_,  from_:to_])), vmin=min, vmax=max,cmap='jet')
        plt.sca( axs[4, i])
        plt.xticks([])
        plt.yticks([])
        axs[4, i].set_aspect('equal')
        if i == 0:
            plt.ylabel('GT')
        im2 = axs[4, 7 * 1 + i].imshow(((img_gt[slice, from_:to_, from_:to_]-img_gt[slice,from_:to_,  from_:to_])), vmin=-max, vmax=max, cmap='jet')
        plt.sca(axs[4, 7 * 1 + i])
        plt.xticks([])
        plt.yticks([])
        axs[4, 7 * 1 + i].set_aspect('equal')
    # =========================
    fig.colorbar(im, ax=axs[4, 0:7], shrink=0.6,extend="both", location='bottom')
    fig.colorbar(im2, ax=axs[4, 6:15], shrink=0.6,extend="both", location='bottom')


    path = parent_path+'fig_miccai2019/'
    fig_nm = path + type + 'slice' + str(slice)
    plt.savefig(fig_nm + '.eps', format='eps', dpi=1000)

    print(fig_nm)
    # fig.colorbar(im, ax=axs[3, 0:7], shrink=1.0,pad=0.0,extend="both", location='bottom', fraction=.1)
    # fig.colorbar(im2, ax=axs[3, 7:14], shrink=1.0, pad=0.0,extend="both",location='bottom', fraction=.1)
    plt.subplots_adjust(wspace=.0, hspace=.0)
    # plt.tight_layout()
    plt.show()