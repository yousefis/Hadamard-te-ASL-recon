import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/debug_Log/synth-8-shark/results/41_PP5_18_cinema5/'

fig=plt.figure(figsize=(8, 4))
columns = 7
rows = 2
for i in range(7):
    # I= sitk.GetArrayFromImage(sitk.ReadImage(path+'angi_'+str(i)+'.mha'))
    # gt= sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/angio_'+str(i)+'.mha'))
    I = sitk.GetArrayFromImage(sitk.ReadImage(path + 'perf_' + str(i) + '.mha'))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(path + 'GT/perf_' + str(i) + '.mha'))
    sliceNo=41
    img=np.rot90(np.rot90(I[sliceNo,:,:]))
    img[np.where(img<.03)]=0
    img2=np.rot90(np.rot90(gt[sliceNo,:,:]))
    ax=fig.add_subplot(rows, columns, i+8)
    # vmax=100
    # vmin=0.001
    plt.imshow(img, cmap='gray')#,vmin=vmin,vmax=vmax)
    ax2=fig.add_subplot(rows, columns, i+1)


    plt.imshow(img2,cmap='gray')#vmin=vmin,vmax=vmax)
    ax.axis('off')
    ax2.axis('off')
    fig.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()