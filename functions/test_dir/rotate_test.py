from functions.image_reader.rotate import rotate_image
import SimpleITK as sitk
import functions.test_dir.my_show
import matplotlib.pyplot as plt
II=(sitk.ReadImage('/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/04_PP2_04_cinema1/decoded_non_crush/noncrush_6.nii'))

ri=rotate_image()
I=sitk.GetArrayFromImage(ri.rotate(II,10))
plt.imshow(I[15,:,:],cmap='gray')
plt.figure()
plt.imshow(sitk.GetArrayFromImage(II)[15,:,:],cmap='gray')
plt.show()