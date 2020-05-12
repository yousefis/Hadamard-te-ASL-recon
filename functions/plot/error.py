import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__=='__main__':
    parent_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
    ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
    perceptual_path='Log_perceptual/regularization/perceptual-0/results/'
    multistage_path='multi_stage/experiment-2/results/'
    subject = '18_PP5_53_cinema2/'


    from_ = 20
    to_ = 80
    slice = 49
    max=5#np.max(img_arry)
    min=-5#np.min(img_arry)
    fig = plt.figure()
    type='perf_'
    # ============================
    flag=True
    # ============================
    plt.rcParams.update({'font.size': 18})

    for i in range(0,7):
        scan = type+str(i)+'.mha'
        ssim_img_path = parent_path + ssim_path + subject + scan
        perceptual_img_path = parent_path + perceptual_path + subject + scan
        multistage_img_path = parent_path + multistage_path + subject + scan
        gt_path = parent_path + ssim_path + subject + 'GT/'+type + str(i)+'.mha'

        img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
        img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))-img_gt
        img_perceptual = sitk.GetArrayFromImage(sitk.ReadImage(perceptual_img_path))-img_gt
        img_multistage = sitk.GetArrayFromImage(sitk.ReadImage(multistage_img_path))-img_gt


        img_arry = [img_gt, img_ssim, img_perceptual]

        ax1 = fig.add_subplot(3,7,i+1)
        divider = make_axes_locatable(ax1)
        im1 = ax1.imshow(np.rot90(np.rot90(img_ssim[from_:to_, slice, from_:to_])), vmin=min, vmax=max, cmap='jet')
        if i==0:
            plt.ylabel('SSIM')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        cb=fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.outline.set_visible(False)
        if i is not 6:
            cb.set_ticks([])

        # ============================
        ax1 = fig.add_subplot(3,7,i+8)
        divider = make_axes_locatable(ax1)
        im1 = ax1.imshow(np.rot90(np.rot90(img_perceptual[from_:to_, slice, from_:to_])), vmin=min, vmax=max, cmap='jet')
        if i==0:
            plt.ylabel('Perceptual')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb=fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.outline.set_visible(False)
        if i is not 6:
            cb.set_ticks([])
        # ============================
        ax1 = fig.add_subplot(3, 7, i+15)
        divider = make_axes_locatable(ax1)
        im1 = ax1.imshow(np.rot90(np.rot90(img_multistage[from_:to_, slice, from_:to_])), vmin=min, vmax=max, cmap='jet')
        if i==0:
            plt.ylabel('Multilevel')
        plt.xticks([])
        plt.yticks([])
        plt.subplots_adjust(wspace=0, hspace=0)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cb=fig.colorbar(im1, cax=cax, orientation='vertical')
        cb.outline.set_visible(False)
        if i is not 6:
            cb.set_ticks([])
        # ============================
        # ax2 = fig.add_subplot(4, 7, i+22)
        # im2 = ax2.imshow(np.rot90(np.rot90(img_gt[from_:to_, slice, from_:to_])), vmin=min, vmax=max, cmap='jet')
        # if i==0:
        #     plt.ylabel('Ground truth')
        # plt.xticks([])
        # plt.yticks([])
        # plt.subplots_adjust(wspace=0, hspace=0)
        # divider = make_axes_locatable(ax2)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # cb=fig.colorbar(im2, cax=cax, orientation='vertical')
        # cb.outline.set_visible(False)
        # if i is not 6:
        #     cb.set_ticks([])

        flag=False
    # plt.title('Time points')
    path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/fig_miccai2019/'
    fig_nm=path+type+'error_slice'+str(slice)
    plt.savefig(fig_nm+'.eps', format='eps', dpi=1000)
    print(fig_nm)
    plt.show()
