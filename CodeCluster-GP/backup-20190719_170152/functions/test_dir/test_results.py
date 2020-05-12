import  SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
path='/exports/lkeb-hpc/syousefi/Data/Simulated_data/BrainWeb_permutation7/04_PP2_04_cinema1/decoded_crush/'
I1=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_0.nii'))
I2=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_1.nii'))
I3=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_2.nii'))
I4=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_3.nii'))
I5=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_4.nii'))
I6=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_5.nii'))
I7=sitk.GetArrayFromImage(sitk.ReadImage(path+'crush_6.nii'))

x =43
y =90
z =144

pix=[I1[x,y,z],I2[x,y,z],I3[x,y,z],I4[x,y,z],I5[x,y,z],I6[x,y,z],I7[x,y,z]]

#
# gt1=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_0.mha'))
# gt2=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_1.mha'))
# gt3=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_2.mha'))
# gt4=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_3.mha'))
# gt5=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_4.mha'))
# gt6=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_5.mha'))
# gt7=sitk.GetArrayFromImage(sitk.ReadImage(path+'GT/perf_6.mha'))
# pix2=[gt1[x,y,z],gt2[x,y,z],gt3[x,y,z],gt4[x,y,z],gt5[x,y,z],gt6[x,y,z],gt7[x,y,z]]
# plt.imshow(I1[z,:,:])

plt.figure()
plt.plot(pix,'g')
plt.show()# plt.plot(pix2,'r')
