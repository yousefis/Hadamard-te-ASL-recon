import tensorflow as tf
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from functions.test_dir.my_show import myshow
class rotate_image:
    def resample(self,image, transform):
        reference_image = image
        interpolator = sitk.sitkBSpline
        default_value = 0
        return sitk.Resample(image, reference_image, transform,
                             interpolator, default_value)
    def rotate(self,image,degrees):

        affine = sitk.AffineTransform(3)
        radians = np.pi * degrees / 180.
        affine.Rotate(axis1=0, axis2=1, angle=radians)
        resampled = self.resample(image, affine)
        # plt.imshow(sitk.GetArrayFromImage(img)[40,:,:])
        # plt.figure()
        # plt.imshow(sitk.GetArrayFromImage(resampled)[40,:,:])
        # plt.show()
        return resampled

