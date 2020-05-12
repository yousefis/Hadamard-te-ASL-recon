from functions.image_reader.read_data import _read_data
import  SimpleITK as sitk
import numpy as np
def max_min_data(train_data,M=10E-5,m=10E5,tissue=1):

    for i in range(len(train_data)):
        for j in range(len(train_data[i][2])):
            if tissue==1:
                I0=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][0][j]))
                I1=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][1][j]))
                I2=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                I3=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][3][j]))
                I4=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][4][j]))
                I5=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][5][j]))
                I6=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][6][j]))
                I7=sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][7][j]))
                r0=np.where(I0)/np.where(not I0)
                r1=np.where(I1)/np.where(not I1)
                r2=np.where(I2)/np.where(not I2)
                r3=np.where(I3)/np.where(not I3)
                r4=np.where(I4)/np.where(not I4)
                r5=np.where(I5)/np.where(not I5)
                r6=np.where(I6)/np.where(not I6)
                r7=np.where(I7)/np.where(not I7)
            else:#angio
                I0 = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][2][j]))
                I1 = sitk.GetArrayFromImage(sitk.ReadImage(train_data[i][1][j]))

                r0 = np.where(I0) / np.where(not I0)
                r1 = np.where(I1) / np.where(not I1)
                r2 = np.where(I2) / np.where(not I2)
                r3 = np.where(I3) / np.where(not I3)
                r4 = np.where(I4) / np.where(not I4)
                r5 = np.where(I5) / np.where(not I5)
                r6 = np.where(I6) / np.where(not I6)
                r7 = np.where(I7) / np.where(not I7)
    return m,M