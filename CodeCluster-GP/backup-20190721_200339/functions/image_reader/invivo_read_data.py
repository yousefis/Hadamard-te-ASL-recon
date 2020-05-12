import tensorflow as tf
import SimpleITK as sitk
# import math as math
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from random import shuffle
# import matplotlib.pyplot as plt


class _invivo_read_data:
    def __init__(self, data,reverse, train_tag='', validation_tag='',
                 test_tag='',
                 img_name='.mha', label_name='', torso_tag='',dataset_path=''):
        Server='DL'
        path='/exports/lkeb-hpc/syousefi/Data/invivo_decomposed/'



        self.train_image_path = path + train_tag
        self.validation_image_path = path + validation_tag
        self.test_image_path = path + test_tag

        self.img_name = img_name
        self.label_name = label_name
        self.data = data

        self.resampled_path = dataset_path
        self.seed=1
        self.reverse=reverse



    # ========================
    def read_data_nonsmooth(self, data_dir):
        image_path = self.resampled_path
        subjects=[]

        for pd in data_dir:
            # print('========================')
            # print(pd)
            # print('========================')
            img_no = pd.split('/')[-1].split('_')[-2]

            subject=[]
            crush = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    isfile(join(image_path, pd, dt))and dt.startswith('crush')]
            crush.sort()#reverse=self.reverse)
            non_crush = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                    isfile(join(image_path, pd, dt))and dt.startswith('nocrush')and not dt.startswith('sig__smooth')]
            non_crush.sort()#reverse=self.reverse)
            crush_noncrush_input=[]
            for i in range(0,8,2):
                crush_noncrush_input.append(crush[i])
                crush_noncrush_input.append(non_crush[i+1])

            perfusion_path=[join(image_path, pd,'decoded_crush', dt) for dt in listdir(join(image_path, pd, 'decoded_crush/'))if
                     not dt.startswith('crush__smooth')]
            perfusion_path.sort()

            angio_path=[join(image_path, pd,'decoded_non_crush', dt) for dt in listdir(join(image_path, pd, 'decoded_non_crush/'))if
                     not dt.startswith('noncrush__smooth')]
            angio_path.sort()
            subject.append(crush)
            subject.append(non_crush)
            subject.append(perfusion_path)
            subject.append(angio_path)
            # subject.append(self.atlas_path+img_no+'/subject'+str(img_no)+'_t1w_p4_brain.nii')
            # subject.append(self.atlas_path+img_no+'/subject'+str(img_no)+'_t1w_p4_brain_seg.nii')
            subjects.append(subject)
        return subjects

    # ========================
    def read_data_path(self):  # join(self.resampled_path, f)
        if np.shape(self.resampled_path)==(2,):
            #read from two paths:
            all_dir=[[join(self.resampled_path[i], f)
              for f in listdir(self.resampled_path[i])
              if (not (isfile(join(self.resampled_path[i], f))) and
                  not (os.path.isfile(join(self.resampled_path[i], f + '/delete.txt'))))]
                     for i in range(len(self.resampled_path))]
            data_dir=np.concatenate(all_dir).ravel()
        else:
            data_dir = [join(self.resampled_path, f)
                    for f in listdir(self.resampled_path)
                    if (not (isfile(join(self.resampled_path, f))) and
                        not (os.path.isfile(join(self.resampled_path, f + '/delete.txt'))))]
        data_dir = np.sort(data_dir)



        test_data = self.read_data_nonsmooth(
            np.hstack((data_dir[:])))


        return test_data



    # ========================
    def read_volume(self, path):
        ct = sitk.ReadImage(path)
        voxel_size = ct.GetSpacing()
        origin = ct.GetOrigin()
        direction = ct.GetDirection()
        ct = sitk.GetArrayFromImage(ct)
        return ct, voxel_size, origin, direction

    # ========================
    def read_imape_path(self, image_path):
        CTs = []
        GTVs = []
        Torsos = []
        data_dir = [join(image_path, f) for f in listdir(image_path) if ~isfile(join(image_path, f))]
        if self.data == 1:
            for pd in data_dir:
                date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                        ~isfile(join(image_path, pd, dt))]
                for dt in date:
                    CT_path = [(join(image_path, pd, dt, self.prostate_ext_img, f)) for f in
                               listdir(join(image_path, pd, dt, self.prostate_ext_img)) if
                               f.endswith(self.img_name)]
                    GTV_path = [join(image_path, pd, dt, self.prostate_ext_gt, f) for f in
                                listdir(join(image_path, pd, dt, self.prostate_ext_gt)) if
                                f.endswith(self.label_name)]

                    CTs.append(CT_path)
                    GTVs.append(GTV_path)

            return CTs, GTVs, Torsos
        elif self.data == 2:
            for pd in data_dir:
                date = [join(image_path, pd, dt) for dt in listdir(join(image_path, pd)) if
                        ~isfile(join(image_path, pd, dt))]

                for dt in date:

                    # read CT and GTV images
                    CT_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                               f.startswith(self.img_name)]
                    GTV_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                f.endswith(self.label_name)]
                    Torso_path = [join(image_path, pd, dt, f) for f in listdir(join(image_path, pd, dt)) if
                                  f.endswith(self.torso_tag)]

                    # print('%s\n%s\n%s' % (
                    # CT_path[len(GTV_path) - 1], GTV_path[len(GTV_path) - 1], Torso_path[len(GTV_path) - 1]))

                    CT4D_path = [(join(image_path, pd, dt, f)) for f in listdir(join(image_path, pd, dt)) if
                                 f.startswith(self.startwith_4DCT) & f.endswith('_padded.mha')]
                    CT_path = CT_path + CT4D_path  # here sahar
                    for i in range(len(CT4D_path)):
                        name_gtv4d = 'GTV_4DCT_' + CT4D_path[i].split('/')[10].split('.')[0].split('_')[1] + '_padded.mha'
                        GTV_path.append((str(join(image_path, pd, dt, name_gtv4d))))  # here sahar
                        Torso_gtv4d = CT4D_path[i].split('/')[10].split('.')[0] + '_Torso.mha'
                        Torso_path.append((str(join(image_path, pd, dt, Torso_gtv4d))))

                        # print('%s\n%s\n%s'%(CT_path[len(GTV_path)-1],GTV_path[len(GTV_path)-1],Torso_path[len(GTV_path)-1]))

                    CTs += (CT_path)
                    GTVs += (GTV_path)
                    Torsos += (Torso_path)

            return CTs, GTVs, Torsos



    def return_depth_width_height(self, CTs):
        CT_image = sitk.ReadImage(''.join(CTs[int(0)]))
        CT_image = sitk.GetArrayFromImage(CT_image)
        return CT_image.shape[0], CT_image.shape[1], CT_image.shape[2]

    def return_normal_const(self, CTs):
        min_normal = 1E+10
        max_normal = -min_normal

        for i in range(len(CTs)):
            CT_image = sitk.ReadImage(''.join(CTs[int(i)]))
            CT_image = sitk.GetArrayFromImage(CT_image)
            max_tmp = np.max(CT_image)
            if max_tmp > max_normal:
                max_normal = max_tmp
            min_tmp = np.min(CT_image)
            if min_tmp < min_normal:
                min_normal = min_tmp
        return min_normal, max_normal

    # =================================================================
    def image_padding(self, img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img


    # =================================================================
    def read_image(self, CT_image, GTV_image, img_height, img_padded_size, seg_size, depth):
        img = CT_image[depth, 0:img_height - 1, 0:img_height - 1]
        img1 = np.zeros((1, img_padded_size, img_padded_size))
        fill_val = img[0][0]
        img1[0][:][:] = np.lib.pad(img, (
            int((img_padded_size - img_height) / 2 + 1), int((img_padded_size - img_height) / 2 + 1)),
                                   "constant", constant_values=(fill_val, fill_val))
        img = img1[..., np.newaxis]
        seg1 = (GTV_image[depth, int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2),
                int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2)])
        seg = np.eye(2)[seg1]
        seg = seg[np.newaxis]
        return img, seg

    # =================================================================
    def check(self, GTVs, width_patch, height_patch, depth_patch):
        no_of_images = len(GTVs)
        for ii in range(no_of_images):
            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            if (max(depth_patch[ii]) > len(GTV_image)):
                print('error')

    # =================================================================

    def shuffle_lists(self, rand_width1, rand_height1, rand_depth1):
        index_shuf = list(range(len(rand_width1)))
        shuffle(index_shuf)
        rand_width11 = np.hstack([rand_width1[sn]]
                                 for sn in index_shuf)
        rand_depth11 = np.hstack([rand_depth1[sn]]
                                 for sn in index_shuf)
        rand_height11 = np.hstack([rand_height1[sn]]
                                  for sn in index_shuf)
        return rand_width11, rand_height11, rand_depth11



    # =================================================================

    def read_all_validation_batches(self, CTs, GTVs, total_sample_no, GTV_patchs_size, patch_window, img_width,
                                    img_height, epoch, img_padded_size, seg_size, whole_image=0):
        self.seed += 1
        np.random.seed(self.seed)

        if whole_image:
            # img_padded_size = 519
            # seg_size = 505

            ii = np.random.randint(0, len(CTs), size=1)
            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))

            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image)  # /CT_image.mean()

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            GTV_max = GTV_image.max()

            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])



            rand_depth = np.random.randint(tumor_begin, tumor_end + 1, size=1)

            img = CT_image[rand_depth, 0:img_height - 1, 0:img_height - 1]

            img1 = np.zeros((1, img_padded_size, img_padded_size))
            fill_val = img[0][0][0]
            img1[0][:][:] = np.lib.pad(img[0], (
                int((img_padded_size - img_height) / 2 + 1), int((img_padded_size - img_height) / 2 + 1)),
                                       "constant", constant_values=(fill_val, fill_val))
            img = img1[..., np.newaxis]

            seg1 = (
            GTV_image[rand_depth, int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2),
            int(img_height / 2) - int(seg_size / 2) - 1:int(img_height / 2) + int(seg_size / 2)])
            seg = np.eye(2)[seg1]
            return img, seg


        else:
            print("Reading %d Validation batches... " % (total_sample_no))

            len_CT = len(CTs)  # patients number
            sample_no = int(total_sample_no / len_CT)  # no of samples that must be selected from each patient
            if sample_no * len_CT < total_sample_no:  # if division causes to reduce total samples
                remain = total_sample_no - sample_no * len_CT
                sample_no = sample_no + remain

            CT_image_patchs = []
            GTV_patchs = []
            for ii in range(len_CT):  # select samples from each patient:

                GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
                GTV_image = sitk.GetArrayFromImage(GTV_image)
                tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
                tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

                CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
                CT_image = sitk.GetArrayFromImage(CT_image1)
                CT_image = (CT_image)  # /CT_image.mean()
                '''random numbers for selecting random samples'''
                rand_depth = np.random.randint(0, len(GTV_image),
                                               size=int(sample_no / 2))  # get half depth samples from every where
                rand_width = np.random.randint(int(patch_window / 2) + 1, img_width - int(patch_window / 2),
                                               size=int(sample_no / 2))  # half of width samples
                rand_height = np.random.randint(int(patch_window / 2) + 1, img_height - int(patch_window / 2),
                                                size=int(sample_no / 2))  # half of width samples
                # print('0')

                '''balencing the classes:'''
                counter = 0
                rand_depth1 = rand_depth  # depth sequence
                rand_width1 = rand_width  # width sequence
                rand_height1 = rand_height  # heigh sequence
                while counter < int(sample_no / 2):  # select half of samples from tumor only!
                    # print("counter: %d" %(counter))
                    dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1)  # select one slice
                    ones = np.where(GTV_image[dpth, 0:img_width,
                                    0:img_height] != 0)  # GTV indices of slice which belong to tumor
                    if len(ones[0]):  # if not empty
                        tmp = int((sample_no * .5) / (tumor_end - tumor_begin))
                        if tmp:
                            rnd_ones = np.random.randint(0, len(ones[0]),
                                                         size=tmp)  # number of samples from each slice
                        else:
                            rnd_ones = np.random.randint(0, len(ones[0]),
                                                         size=1)  # number of samples from each slice

                        counter += len(rnd_ones)  # counter for total samples
                        rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                        rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))
                        rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))

                # print('1')
                GTV_max = GTV_image.max()

                CT_image_patchs1 = np.stack([(CT_image[int(rand_depth1[sn]),
                                              int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[
                                                                                                        sn]) + int(
                                                  patch_window / 2),
                                              int(rand_height1[sn]) - int(patch_window / 2) - 1: int(
                                                  rand_height1[sn]) +
                                                                                                 int(
                                                                                                     patch_window / 2)])[
                                                 ..., np.newaxis]
                                             for sn in range(len(rand_height1))])
                # print('2')

                GTV_patchs1 = np.stack([(GTV_image[
                                         int(rand_depth1[sn]),
                                         int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                         int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                         , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                         int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                         ]).astype(int)
                                        for sn in
                                        range(len(rand_height1))]).reshape(len(rand_height1), GTV_patchs_size,
                                                                           GTV_patchs_size)
                # GTV_patchs1=int(GTV_patchs1/GTV_image.max())

                # print('3')

                # print("GTV_patchs min: %d, max: %d  tumor sample no: %d" % (GTV_patchs1.min(),
                #                                                             GTV_patchs1.max(),
                #                                                             len(np.where(GTV_patchs1 != 0)[0])
                #                                                             ))
                GTV_patchs1 = np.eye(2)[GTV_patchs1]
                # print('4')

                if len(CT_image_patchs):
                    CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs1))
                    GTV_patchs = np.vstack((GTV_patchs, GTV_patchs1))
                else:
                    CT_image_patchs = CT_image_patchs1
                    GTV_patchs = GTV_patchs1

                    # print('5')

                    # print("length: %d" % (len(CT_image_patchs)))

            '''remove the further samples'''
            if len(CT_image_patchs) > total_sample_no:
                CT_image_patchs = np.delete(CT_image_patchs, list(range(total_sample_no, len(CT_image_patchs))), 0)
                GTV_patchs = np.delete(GTV_patchs, list(range(total_sample_no, len(GTV_patchs))), 0)

            '''shuffle the lists'''
            index_shuf = list(range(len(CT_image_patchs)))
            shuffle(index_shuf)
            GTV_patchs1 = np.vstack([GTV_patchs[sn][:][:]]
                                    for sn in index_shuf)
            CT_image_patchs1 = np.vstack([CT_image_patchs[sn][:][:]]
                                         for sn in index_shuf)

        return CT_image_patchs1, GTV_patchs1

    # =================================================================


    def read_data_all_train_batches(self, CTs, GTVs, total_sample_no, GTV_patchs_size, patch_window, img_width,
                                    img_height, epoch):
        self.seed+=1
        np.random.seed(self.seed)
        print("Reading %d training batches... " % (total_sample_no))
        len_CT = len(CTs)  # patients number
        sample_no = int(total_sample_no / len_CT)  # no of samples that must be selected from each patient
        if sample_no * len_CT < total_sample_no:  # if division causes to reduce total samples
            # remain=total_sample_no-sample_no*len_CT
            sample_no = sample_no + 2

        CT_image_patchs = []
        GTV_patchs = []
        for ii in range(len_CT):  # select samples from each patient:

            GTV_image = sitk.ReadImage(''.join(GTVs[int(ii)]))
            GTV_image = sitk.GetArrayFromImage(GTV_image)
            tumor_begin = np.min(np.where(GTV_image != GTV_image[0][0][0])[0])
            tumor_end = np.max(np.where(GTV_image != GTV_image[0][0][0])[0])

            CT_image1 = sitk.ReadImage(''.join(CTs[int(ii)]))
            CT_image = sitk.GetArrayFromImage(CT_image1)
            CT_image = (CT_image)  # /CT_image.mean()
            '''random numbers for selecting random samples'''
            rand_depth = np.random.randint(0, len(GTV_image),
                                           size=int(sample_no / 2))  # get half depth samples from every where
            rand_width = np.random.randint(int(patch_window / 2) + 1, img_width - int(patch_window / 2),
                                           size=int(sample_no / 2))  # half of width samples
            rand_height = np.random.randint(int(patch_window / 2) + 1, img_height - int(patch_window / 2),
                                            size=int(sample_no / 2))  # half of width samples
            # print('0')

            '''balencing the classes:'''
            counter = 0
            rand_depth1 = rand_depth  # depth sequence
            rand_width1 = rand_width  # width sequence
            rand_height1 = rand_height  # heigh sequence
            while counter < int(sample_no / 2):  # select half of samples from tumor only!
                # print("c
                # ounter: %d" %(counter))
                dpth = np.random.randint(tumor_begin, tumor_end + 1, size=1)  # select one slice
                ones = np.where(
                    GTV_image[dpth, 0:img_width, 0:img_height] != 0)  # GTV indices of slice which belong to tumor
                if len(ones[0]):  # if not empty
                    tmp = int((sample_no * .5) / (tumor_end - tumor_begin))
                    if tmp:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=tmp)  # number of samples from each slice
                    else:
                        rnd_ones = np.random.randint(0, len(ones[0]), size=1)  # number of samples from each slice

                    counter += len(rnd_ones)  # counter for total samples
                    rand_width1 = np.hstack((rand_width1, ones[1][rnd_ones]))
                    rand_height1 = np.hstack((rand_height1, ones[2][rnd_ones]))
                    rand_depth1 = np.hstack((rand_depth1, dpth * np.ones(len(rnd_ones))))

            # print('1')
            GTV_max = GTV_image.max()

            CT_image_patchs1 = np.stack([(CT_image[int(rand_depth1[sn]),
                                          int(rand_width1[sn]) - int(patch_window / 2) - 1: int(rand_width1[
                                                                                                    sn]) + int(
                                              patch_window / 2),
                                          int(rand_height1[sn]) - int(patch_window / 2) - 1: int(rand_height1[sn]) +
                                                                                             int(patch_window / 2)])[
                                             ..., np.newaxis]
                                         for sn in range(len(rand_height1))])
            # print('2')

            GTV_patchs1 = np.stack([(GTV_image[
                                     int(rand_depth1[sn]),
                                     int(rand_width1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_width1[sn]) + int(GTV_patchs_size / 2)
                                     , int(rand_height1[sn]) - int(GTV_patchs_size / 2) - 1:
                                     int(rand_height1[sn]) + int(GTV_patchs_size / 2)
                                     ]).astype(int)
                                    for sn in
                                    range(len(rand_height1))]).reshape(len(rand_height1), GTV_patchs_size,
                                                                       GTV_patchs_size)
            # GTV_patchs1=int(GTV_patchs1/GTV_image.max())

            # print('3')

            # print("GTV_patchs min: %d, max: %d  tumor sample no: %d" % (GTV_patchs1.min(),
            #                                                             GTV_patchs1.max(),
            #                                                             len(np.where(GTV_patchs1 != 0)[0])
            #                                                             ))
            GTV_patchs1 = np.eye(2)[GTV_patchs1]
            # print('4')

            if len(CT_image_patchs):
                CT_image_patchs = np.vstack((CT_image_patchs, CT_image_patchs1))
                GTV_patchs = np.vstack((GTV_patchs, GTV_patchs1))
            else:
                CT_image_patchs = CT_image_patchs1
                GTV_patchs = GTV_patchs1

                # print('5')

                # print("length: %d" %(len(CT_image_patchs)))

        '''remove the further samples'''
        if len(CT_image_patchs) > total_sample_no:
            CT_image_patchs = np.delete(CT_image_patchs, list(range(total_sample_no, len(CT_image_patchs))), 0)
            GTV_patchs = np.delete(GTV_patchs, list(range(total_sample_no, len(GTV_patchs))), 0)

        '''shuffle the lists'''
        index_shuf = list(range(len(CT_image_patchs)))
        shuffle(index_shuf)
        GTV_patchs1 = np.vstack([GTV_patchs[sn][:][:]]
                                for sn in index_shuf)
        CT_image_patchs1 = np.vstack([CT_image_patchs[sn][:][:]]
                                     for sn in index_shuf)

        return GTV_patchs1, CT_image_patchs1