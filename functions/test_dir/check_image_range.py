from functions.image_reader.read_data import _read_data
import  SimpleITK as sitk
import numpy as np
def max_min_data(train_data,M=10E-5,m=10E5,which=0):

    for i in range(len(train_data)):
        for j in range(len(train_data[i][2])):
            if which == 0: # ASL crush
                I = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][0][j]))
                M = np.max([np.max(I), M])
                m = np.min([np.min(I), m])

            elif which == 1: # ASL non-crush
                I = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][1][j]))
                M = np.max([np.max(I), M])
                m = np.min([np.min(I), m])

            elif which == 2: # perfusion
                I=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                M=np.max([np.max(I),M])
                m=np.min([np.min(I),m])

            else: # angio
                I = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                I1 = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][3][j]))
                I2 = I1 - I
                M = np.max([np.max(I2), M])
                m = np.min([np.min(I2), m])
    return m,M