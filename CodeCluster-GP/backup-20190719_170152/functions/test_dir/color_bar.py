import SimpleITK as sitk
import matplotlib.pyplot as plt

parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
perceptual_path = 'Log_perceptual/regularization/perceptual-0/results/'
multistage_path = 'multi_stage/experiment-2/results/'
subject = '18_PP5_53_cinema2/'
from_ = 20
to_ = 85
slice = 49
max = 6  # np.max(img_arry)
min = -1  # np.min(img_arry)
i=3
type = 'perf_'
scan = type + str(i) + '.mha'
ssim_img_path = parent_path + ssim_path + subject + scan
img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))
I=img_ssim[slice,from_:to_,  from_:to_]


rectLS=[]
row=8
col=14
w=1/(col)
for x in range(col):
   for y in range(row):
       rectLS.append([x*w, y*w, w,w])



axLS=[]
fig=plt.figure()
for i in range(row*col): #all images
    if i %row==0:
        axLS.append(fig.add_axes(rectLS[i]))
        plt.xticks([])
        plt.yticks([])
for i in range(row*col):
     ax=fig.add_axes(rectLS[i], sharey=axLS[-1])
     ax.imshow(I)
     axLS.append(ax)
     plt.xticks([])
     plt.yticks([])
# axLS.append(fig.add_axes(rectLS[4]))
# plt.xticks([])
# plt.yticks([])
# for i in [1,2,3]:
#      axLS.append(fig.add_axes(rectLS[i+4],sharex=axLS[i],sharey=axLS[-1]))
#      plt.xticks([])
#      plt.yticks([])
# axLS.append(fig.add_axes(rectLS[8]))
# plt.xticks([])
# plt.yticks([])
# for i in [5,6,7]:
#      axLS.append(fig.add_axes(rectLS[i+4],sharex=axLS[i],sharey=axLS[-1]))
#      plt.xticks([])
#      plt.yticks([])
# axLS.append(fig.add_axes(rectLS[12]))
# plt.xticks([])
# plt.yticks([])
# for i in [9,10,11]:
#      axLS.append(fig.add_axes(rectLS[i+4],sharex=axLS[i],sharey=axLS[-1]))
#      plt.xticks([])
#      plt.yticks([])
# plt.xticks([])
# plt.yticks([])
plt.show()
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# import SimpleITK as sitk
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import matplotlib.gridspec as gridspec
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
#
# if __name__=='__main__':
#
#     parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/'
#     ssim_path = 'normal_SSIM_p_a_newdb/experiment-1/results/'
#     perceptual_path = 'Log_perceptual/regularization/perceptual-0/results/'
#     multistage_path = 'multi_stage/experiment-2/results/'
#     subject = '18_PP5_53_cinema2/'
#
#     from_ = 20
#     to_ = 80
#     slice = 49
#     max = 6  # np.max(img_arry)
#     min = -1  # np.min(img_arry)
#     fig = plt.figure()
#     # fig, axs = plt.subplots(nrows=4, ncols=14, figsize=(9, 6),
#     #                         subplot_kw={'xticks': [], 'yticks': []})
#     type = 'perf_'
#     # ============================
#     flag = True
#     # ============================
#     plt.rcParams.update({'font.size': 16})
#
#     fontdicty = {'fontsize': 16,
#                  'weight': 'bold',
#                  'verticalalignment': 'baseline',
#                  'horizontalalignment': 'center'}
#
#
#
#     fig, axs = plt.subplots(4, 14, figsize=(5,5), constrained_layout=True)
#
#
#     for i in range(0, 7):
#         scan = type + str(i) + '.mha'
#         ssim_img_path = parent_path + ssim_path + subject + scan
#         perceptual_img_path = parent_path + perceptual_path + subject + scan
#         multistage_img_path = parent_path + multistage_path + subject + scan
#         gt_path = parent_path + ssim_path + subject + 'GT/' + type + str(i) + '.mha'
#
#         img_ssim = sitk.GetArrayFromImage(sitk.ReadImage(ssim_img_path))
#         img_perceptual = sitk.GetArrayFromImage(sitk.ReadImage(perceptual_img_path))
#         img_multistage = sitk.GetArrayFromImage(sitk.ReadImage(multistage_img_path))
#         img_gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
#
#         im = axs[0,i].imshow(np.rot90(np.rot90(img_ssim[from_:to_, slice, from_:to_])), vmin=min, vmax=max, cmap='jet')
#         plt.sca(axs[0,  i])
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel('SSIM')
#         im2 = axs[0,7*1+i].imshow(np.rot90(np.rot90(img_ssim[from_:to_, slice, from_:to_] - img_gt[from_:to_, slice, from_:to_])),vmin=-max, vmax=max, cmap='jet')
#         plt.sca(axs[0, 7*1+i])
#         plt.xticks([])
#         plt.yticks([])
#
#
#         im = axs[1,i].imshow(np.rot90(np.rot90(img_perceptual[from_:to_, slice, from_:to_])), vmin=min, vmax=max,cmap='jet')
#         plt.sca(axs[1,  i])
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel('Perceptual')
#         im2 = axs[1,7*1+i].imshow(np.rot90(np.rot90(img_perceptual[from_:to_, slice, from_:to_]- img_gt[from_:to_, slice, from_:to_])), vmin=-max, vmax=max,cmap='jet')
#         plt.sca(axs[1, 7 * 1 + i])
#         plt.xticks([])
#         plt.yticks([])
#
#
#         im = axs[2, i].imshow(np.rot90(np.rot90(img_multistage[from_:to_, slice, from_:to_])), vmin=min, vmax=max,cmap='jet')
#         plt.sca(axs[2, i])
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel('Multi-level')
#         im2 = axs[2, 7 * 1 + i].imshow(np.rot90(np.rot90(img_multistage[from_:to_, slice, from_:to_] - img_gt[from_:to_, slice, from_:to_])),vmin=-max, vmax=max, cmap='jet')
#         plt.sca(axs[2, 7 * 1 + i])
#         plt.xticks([])
#         plt.yticks([])
#
#
#         im = axs[3, i].imshow(np.rot90(np.rot90(img_gt[from_:to_, slice, from_:to_])), vmin=min, vmax=max,cmap='jet')
#         plt.sca( axs[3, i])
#         plt.xticks([])
#         plt.yticks([])
#         if i == 0:
#             plt.ylabel('GT')
#         im2 = axs[3, 7 * 1 + i].imshow(np.rot90(np.rot90(img_gt[from_:to_, slice, from_:to_]-img_gt[from_:to_, slice, from_:to_])), vmin=-max, vmax=max, cmap='jet')
#         plt.sca(axs[3, 7 * 1 + i])
#         plt.xticks([])
#         plt.yticks([])
#
#     fig.colorbar(im, ax=axs[3, 0:7], shrink=0.6,extend="both", location='bottom')
#     fig.colorbar(im2, ax=axs[3, 7:14], shrink=0.6,extend="both", location='bottom')
#
#
#     path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/fig_miccai2019/'
#     fig_nm = path + type + 'slice' + str(slice)
#     plt.savefig(fig_nm + '.eps', format='eps', dpi=1000)
#
#     print(fig_nm)
#     # fig.colorbar(im, ax=axs[3, 0:7], shrink=1.0,pad=0.0,extend="both", location='bottom', fraction=.1)
#     # fig.colorbar(im2, ax=axs[3, 7:14], shrink=1.0, pad=0.0,extend="both",location='bottom', fraction=.1)
#     # plt.subplots_adjust(wspace=.0, hspace=.0)
#     plt.show()