from functions.image_reader.read_data import _read_data
import  SimpleITK as sitk
import numpy as np
def max_min_data(train_data,M=10E-5,m=10E5,tissue=0):

    for i in range(len(train_data)):
        for j in range(len(train_data[i][2])):
            if tissue==1:
                I=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                M=np.max([np.max(I),M])
                m=np.min([np.max(I),m])
            else:#angio
                I = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                I1 = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][3][j]))
                I2 = I1 - I
                M = np.max([np.max(I2), M])
                m = np.min([np.max(I2), m])
    return m,M