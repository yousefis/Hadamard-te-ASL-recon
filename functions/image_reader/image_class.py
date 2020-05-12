import collections
import itertools
import random
from random import shuffle

import SimpleITK as sitk
import numpy as np

import functions.utils.settings as settings
from functions.image_reader.rotate import rotate_image

# import matplotlib.pyplot as plt
class image_class:
    def __init__(self,data
                 ,bunch_of_images_no,is_training,patch_window,label_patch_size,sample_no_per_bunch,validation_total_sample):
        self.data=data
        self.bunch_of_images_no = bunch_of_images_no
        self.node = collections.namedtuple('node', 'name crush_encoded noncrush_encoded perfusion angio spacing origin direction mri segmentation')
        self.collection=[]
        self.is_training=is_training
        self.patch_window=patch_window
        self.label_patch_size=label_patch_size
        self.random_images=list(range(0,len(data)))
        self.min_image=-1024
        self.max_image=1500
        self.counter_save=0
        self.end_1st_dataset=375
        self.random_data1 = list(range(0, self.end_1st_dataset))
        self.random_data2 = list(range(self.end_1st_dataset, 456))
        self.deform1stdb=0
        self.static_counter_vl=0
        self.seed = 100
        self.sample_no_per_bunch=sample_no_per_bunch
        self.rotate_image=rotate_image()
        self.validation_total_sample=validation_total_sample



    # --------------------------------------------------------------------------------------------------------
    def creat_mask(self,shape):
        torso_mask = np.ones((shape[0] - int(self.patch_window),
                               shape[1] - int(self.patch_window),
                               shape[2] - int(self.patch_window)))

        torso_mask = np.pad(torso_mask, (
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
            (int(self.patch_window / 2) + 1, int(self.patch_window / 2)),
        ),
                                  mode='constant', constant_values=0)
        return torso_mask


    # --------------------------------------------------------------------------------------------------------
    def image_padding(self,img, padLowerBound, padUpperBound, constant):
        filt = sitk.ConstantPadImageFilter()
        padded_img = filt.Execute(img,
                                  padLowerBound,
                                  padUpperBound,
                                  constant)
        return padded_img

    def image_crop(self,img, padLowerBound, padUpperBound):
        crop_filt = sitk.CropImageFilter()
        cropped_img = crop_filt.Execute(img, padLowerBound, padUpperBound)
        return cropped_img

    def apply_deformation(self,img, BCoeff, defaultPixelValue, spacing, origin, direction, interpolator):
        resampler = sitk.ResampleImageFilter()
        resampler.SetTransform(BCoeff)
        resampler.SetDefaultPixelValue(defaultPixelValue)
        resampler.SetReferenceImage(img)  # set input image
        resampler.SetInterpolator(interpolator)  # set interpolation method
        resampler.SetOutputSpacing(spacing)
        resampler.SetOutputOrigin(origin)
        resampler.SetOutputDirection(direction)
        deformedImg = sitk.Resample(img, BCoeff)
        return deformedImg

    def Bspline_distort(self,CT_image, GTV_image, Torso_image, Penalize_image,max_dis=2):
        z_len = CT_image.GetDepth()
        x_len = CT_image.GetHeight()
        gen = self.random_gen(1, max_dis)
        displace_range = list(itertools.islice(gen, 1))[0]
        grid_space = 4
        z_grid=0
        while not z_grid:
            grid_space+=1
            z_grid = int(grid_space * z_len / (x_len))



        spacing = CT_image.GetSpacing()
        origin = CT_image.GetOrigin()
        direction = CT_image.GetDirection()

        BCoeff = sitk.BSplineTransformInitializer(CT_image, [grid_space, grid_space, z_grid],
                                                  order=3)
        # The third parameter for the BSplineTransformInitializer is the spline order It defaults to 3
        displacements = np.random.uniform(-displace_range, displace_range, int(len(BCoeff.GetParameters())))

        Xdisplacements = np.reshape(displacements[0: (grid_space + 3) * (grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])
        Ydisplacements = np.reshape(displacements[
                                    (grid_space + 3) * (grid_space + 3) * (z_grid + 3): 2 * (grid_space + 3) * (
                                    grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])
        Zdisplacements = np.reshape(displacements[
                                    2 * (grid_space + 3) * (grid_space + 3) * (z_grid + 3):3 * (grid_space + 3) * (
                                    grid_space + 3) * (z_grid + 3)],
                                    [(grid_space + 3), (grid_space + 3), (z_grid + 3)])

        displacements = np.hstack((np.reshape(Xdisplacements, -1),
                                   np.reshape(Ydisplacements, -1),
                                   np.reshape(Zdisplacements, -1)))
        BCoeff.SetParameters(displacements)

        # define sampler
        CT_deformed = self.apply_deformation(img=CT_image, BCoeff=BCoeff,
                                        defaultPixelValue=-1024, spacing=spacing,
                                        origin=origin, direction=direction,
                                        interpolator=sitk.sitkBSpline)
        # define sampler for gtv

        GTV_deformed = self.apply_deformation(img=GTV_image, BCoeff=BCoeff,
                                         defaultPixelValue=0, spacing=spacing,
                                         origin=origin, direction=direction,
                                         interpolator=sitk.sitkNearestNeighbor)


        Torso_deformed = self.apply_deformation(img=Torso_image, BCoeff=BCoeff,
                                           defaultPixelValue=0, spacing=spacing,
                                           origin=origin, direction=direction,
                                           interpolator=sitk.sitkNearestNeighbor)


        return CT_deformed, GTV_deformed, Torso_deformed, []
    def Bspline_distort2(self,CT_image, GTV_image, Torso_image, Penalize_image):
        grid_space = 2
        gen = self.random_gen(1, 5)
        displace_range = list(itertools.islice(gen, 1))[0]


        spacing = CT_image.GetSpacing()
        origin = CT_image.GetOrigin()
        direction = CT_image.GetDirection()
        z_len = CT_image.GetDepth()
        x_len = CT_image.GetHeight()
        padd_zero = x_len - z_len

        CT_image1 = self.image_padding(img=CT_image,
                                  padLowerBound=[0, 0, int(padd_zero / 2)],
                                  padUpperBound=[0, 0, int(padd_zero / 2)],
                                  constant=-1024)

        GTV_image1 = self.image_padding(img=GTV_image,
                                   padLowerBound=[0, 0, int(padd_zero / 2)],
                                   padUpperBound=[0, 0, int(padd_zero / 2)],
                                   constant=-1024)

        Torso_image1 = self.image_padding(img=Torso_image,
                                     padLowerBound=[0, 0, int(padd_zero / 2)],
                                     padUpperBound=[0, 0, int(padd_zero / 2)],
                                     constant=-1024)

        # CT_image2 = image_crop(CT_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        # GTV_image2 = image_crop(GTV_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        # Torso_image2 = image_crop(Torso_image1, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])
        #
        # sitk.WriteImage(CT_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/ct2.mha')
        # sitk.WriteImage(GTV_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/gtv2.mha')
        # sitk.WriteImage(Torso_image2, '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/test/torso2.mha')

        # define transform:
        BCoeff = sitk.BSplineTransformInitializer(CT_image1, [grid_space, grid_space, grid_space],
                                                  order=3)
        # The third parameter for the BSplineTransformInitializer is the spline order It defaults to 3
        displacements = np.random.uniform(-displace_range, displace_range, int(len(BCoeff.GetParameters())))
        param_no = np.int(np.ceil(np.power(len(displacements) / 3, 1 / 3)))

        Xdisplacements = np.reshape(displacements[0: param_no * param_no * param_no],
                                    [param_no, param_no, param_no])
        Ydisplacements = np.reshape(
            displacements[param_no * param_no * param_no: 2 * param_no * param_no * param_no],
            [param_no, param_no, param_no])
        Zdisplacements = np.reshape(
            displacements[2 * param_no * param_no * param_no:3 * param_no * param_no * param_no],
            [param_no, param_no, param_no])

        displacements = np.hstack((np.reshape(Xdisplacements, -1),
                                   np.reshape(Ydisplacements, -1),
                                   np.reshape(Zdisplacements, -1)))
        BCoeff.SetParameters(displacements)

        # define sampler
        CT_deformed = self.apply_deformation(img=CT_image1, BCoeff=BCoeff,
                                        defaultPixelValue=-1024, spacing=spacing,
                                        origin=origin, direction=direction,
                                        interpolator=sitk.sitkBSpline)
        CT_deformed = self.image_crop(CT_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])

        GTV_deformed = self.apply_deformation(img=GTV_image1, BCoeff=BCoeff,
                                         defaultPixelValue=0, spacing=spacing,
                                         origin=origin, direction=direction,
                                         interpolator=sitk.sitkNearestNeighbor)
        GTV_deformed = self.image_crop(GTV_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])

        Torso_deformed = self.apply_deformation(img=Torso_image1, BCoeff=BCoeff,
                                           defaultPixelValue=0, spacing=spacing,
                                           origin=origin, direction=direction,
                                           interpolator=sitk.sitkNearestNeighbor)
        Torso_deformed = self.image_crop(Torso_deformed, [0, 0, int(padd_zero / 2)], [0, 0, int(padd_zero / 2)])


        return CT_deformed, GTV_deformed, Torso_deformed, []


    # --------------------------------------------------------------------------------------------------------
    def Flip(self,CT_image, GTV_image, Torso_image):
        TF1=False
        TF2=bool(random.getrandbits(1))
        TF3=bool(random.getrandbits(1))

        CT_image = sitk.Flip(CT_image, [TF1, TF2, TF3])
        GTV_image = sitk.Flip(GTV_image, [TF1, TF2, TF3])
        Torso_image = sitk.Flip(Torso_image, [TF1, TF2, TF3])
        return CT_image, GTV_image, Torso_image
    # --------------------------------------------------------------------------------------------------------
    def HistogramEqualizer(self,CT_image1):
        return CT_image1

    # --------------------------------------------------------------------------------------------------------
    def resampling(self,image,new_spacing):
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                    int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                    int(round(original_size[2] * (original_spacing[2] / new_spacing[2])))]
        resampleSliceFilter = sitk.ResampleImageFilter()

        sitk_isotropic_xslice = resampleSliceFilter.Execute(image, new_size, sitk.Transform(),
                                                            sitk.sitkNearestNeighbor, image.GetOrigin(),
                                                            new_spacing, image.GetDirection(), 0,
                                                            image.GetPixelID())
        return sitk_isotropic_xslice
    # --------------------------------------------------------------------------------------------------------
    def read_resample_image_test(self,img_name,input_size,new_spacing=(3,3,3)):

        img = sitk.ReadImage(''.join(img_name))
        #resample
        img = self.resampling(img, new_spacing)

        pad_size = input_size - img.GetSize()[0]

        img2 = self.image_padding(img=img,
                                  padLowerBound=[int(pad_size / 2), int(pad_size / 2), int(pad_size / 2) + 12],
                                  padUpperBound=[int(pad_size / 2), int(pad_size / 2), int(pad_size / 2) + 13],
                                  constant=0)
        img=img2







        return sitk.GetArrayFromImage(img2)
    # --------------------------------------------------------------------------------------------------------
    def read_resample_image(self,img_name,degree,new_spacing=(3,3,3)):

        img = sitk.ReadImage(''.join(img_name))
        #resample
        img = self.resampling(img, new_spacing)
        #rotate:
        if degree:
            img = self.rotate_image.rotate(image=img, degrees=degree)

        if self.sample_no_per_bunch == self.bunch_of_images_no:
            img2 = self.image_padding(img=img,
                                      padLowerBound=[15, 15, 15 + 12],
                                      padUpperBound=[15, 15, 15 + 13],
                                      constant=0)
        else:
            img2 = self.image_padding(img=img,
                                      padLowerBound=[int(self.patch_window / 2) + 13,
                                                     int(self.patch_window / 2) + 13,
                                                     int(self.patch_window / 2) + 12 + 13],
                                      padUpperBound=[int(self.patch_window / 2) + 13,
                                                     int(self.patch_window / 2) + 13,
                                                     int(self.patch_window / 2) + 13 + 13],
                                      constant=0)



        return sitk.GetArrayFromImage(img2)
    # --------------------------------------------------------------------------------------------------------
    def read_image_group(self,group_name,degree):
        group_img=[]

        for i in range(len(group_name)):
            img=sitk.ReadImage(''.join(group_name[i]))
            if degree:
                img=self.rotate_image.rotate(image=img,degrees=degree)



            if self.sample_no_per_bunch==self.bunch_of_images_no:
                img2 = self.image_padding(img=img,
                                          padLowerBound=[15, 15, 15+ 12],
                                          padUpperBound=[15, 15, 15+ 13],
                                          constant=0)
            else:
                img2=self.image_padding(img=img,
                                   padLowerBound=[int(self.patch_window/2)+13, int(self.patch_window/2)+13, int(self.patch_window/2) + 12+13],
                                   padUpperBound=[int(self.patch_window/2)+13, int(self.patch_window/2)+13, int(self.patch_window/2) + 13+13],
                                   constant=0)



            # imgg = sitk.GetArrayFromImage(img2)
            # plt.imshow(imgg[30, :, :])
            # if deform == 1:
            #     img = self.Bspline_distort(img
            #                                                                                   max_dis=max_dis)

            group_img.append(sitk.GetArrayFromImage(img2))
            if i==0:
                spacing=img.GetSpacing()
                direction=img.GetDirection()
                origin=img.GetOrigin()
        return group_img,spacing,direction,origin
    # --------------------------------------------------------------------------------------------------------
    def calculate_angio(self,crush_decoded,noncrush_decoded):
        angios = []
        for i in range(len(crush_decoded)):
            img = noncrush_decoded[i]-crush_decoded[i]
            img[np.where(img<0)]=0
            angios.append(img)
        return angios
    # --------------------------------------------------------------------------------------------------------
    def read_image_group_for_test(self,group_name,input_size,final_layer=0):
        group_img=[]

        for i in range(len(group_name)):
            img=sitk.ReadImage(''.join(group_name[i]))
            pad_size=input_size-img.GetSize()[0]


            img2 = self.image_padding(img=img,
                  padLowerBound=[int(pad_size/2), int(pad_size/2), int(pad_size/2)+ 12],
                  padUpperBound=[int(pad_size/2), int(pad_size/2), int(pad_size/2)+ 13],
                  constant=0)
            img2=sitk.GetArrayFromImage(img2)
            if final_layer is not 0:
                img2=img2[int((np.shape(img2)[0] - final_layer)/ 2) : int((np.shape(img2)[0] + final_layer)/ 2),
                          int((np.shape(img2)[0] - final_layer) / 2) : int((np.shape(img2)[0] + final_layer) / 2),
                          int((np.shape(img2)[0] - final_layer) / 2) : int((np.shape(img2)[0] + final_layer) / 2),]

            group_img.append(img2)
            if i==0:
                spacing=img.GetSpacing()
                direction=img.GetDirection()
                origin=img.GetOrigin()
        return group_img,spacing,direction,origin
    # --------------------------------------------------------------------------------------------------------
    def read_image_for_test(self,test_set, img_indx,input_size,final_layer,invivo=0):
        crush_encoded, spacing, direction, origin = self.read_image_group_for_test(test_set[int(img_indx)][0],input_size=input_size)
        noncrush_encoded, spacing, direction, origin = self.read_image_group_for_test(test_set[int(img_indx)][1],input_size=input_size)
        crush_decoded, spacing, direction, origin = self.read_image_group_for_test(test_set[int(img_indx)][2],input_size=input_size,final_layer=final_layer)
        noncrush_decoded, spacing, direction, origin = self.read_image_group_for_test(test_set[int(img_indx)][3],input_size=input_size,final_layer=final_layer)
        if invivo==0:
            mri_decoded = self.read_resample_image_test( self.data[int(img_indx)][4],input_size=input_size)
            segmentation_decoded = self.read_resample_image( self.data[int(img_indx)][5],degree=0)
        else:
            mri_decoded=None
            segmentation_decoded=None

        angio = self.calculate_angio(crush_decoded=crush_decoded, noncrush_decoded=noncrush_decoded)
        for i in range(len(crush_decoded)):
            crush_decoded[i][:, :, :] = crush_decoded[i][:, :, :]

        return [crush_encoded,noncrush_encoded,crush_decoded,angio,mri_decoded,segmentation_decoded,spacing, direction, origin]
    # --------------------------------------------------------------------------------------------------------
    #read information of each image
    def read_image(self,img_index,deform,max_dis=0,tr=1):
        degree=0
        # if tr:
            # if np.random.randint(0, 10, 1) >= 7:
            #     degree = np.random.randint(-13, 13, 1)[0]
            #     print('rrrr')

        crush_encoded,spacing,direction,origin=self.read_image_group( self.data[int(img_index)][0],degree=degree)
        noncrush_encoded,spacing,direction,origin=self.read_image_group( self.data[int(img_index)][1],degree=degree)
        crush_decoded,spacing,direction,origin=self.read_image_group( self.data[int(img_index)][2],degree=degree)
        noncrush_decoded,spacing,direction,origin=self.read_image_group( self.data[int(img_index)][3],degree=degree)
        brain=self.read_resample_image( self.data[int(img_index)][4],degree=degree)
        segmentation=self.read_resample_image( self.data[int(img_index)][5],degree=degree)
        angio=self.calculate_angio(crush_decoded=crush_decoded,noncrush_decoded=noncrush_decoded)
        for i in range(len(crush_decoded)):
            crush_decoded[i][:, :, :] = crush_decoded[i][:, :, :]
        n = self.node(name=self.data[int(img_index)][0],crush_encoded=crush_encoded, noncrush_encoded=noncrush_encoded, perfusion=crush_decoded, angio=angio,
                      spacing=spacing, origin=origin, direction=direction,mri=brain, segmentation=segmentation)
        # plt.imshow(angio[0][30, :, :])
        return n

    def return_normal_image(self,CT_image,max_range,min_range,min_normal,max_normal):
        return (max_range - min_range) * (
        (CT_image - min_normal) / (max_normal - min_normal)) + min_range
    # --------------------------------------------------------------------------------------------------------
    def random_gen(self,low, high):
        while True:
            yield random.randrange(low, high)
    # --------------------------------------------------------------------------------------------------------
    def read_bunch_of_images_from_dataset_vl(self,is_training):  # for training
        if len(settings.subjects_vl_segmentation) > self.validation_total_sample:
            return
        if settings.vl_isread == False:
            return

        settings.read_patche_mutex_vl.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)

        if len(self.random_images) < self.bunch_of_images_no:  # if there are no image choices for selection
            self.random_images = list(range(0, len(self.data)))
            self.deform1stdb+=1
        # select some distinct images for extracting patches!
        rand_image_no = np.random.randint(0, 10,self.bunch_of_images_no)


        self.random_images = [x for x in range(len(self.random_images)) if
                              x not in rand_image_no]  # remove selected images from the choice list
        print(rand_image_no)

        for img_index in range(len(rand_image_no)):
            if len(settings.subjects_vl_segmentation) > self.validation_total_sample:
                self.collection.clear()
                return
            deform = 0
            max_dis=0
            imm = self.read_image(rand_image_no[img_index], deform=deform,max_dis=max_dis,tr=0)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('validation image no read so far: %s'%len(self.collection))
        if is_training == True:
            settings.tr_isread=False
        else:
            settings.vl_isread = False
        settings.read_patche_mutex_vl.release()
    #================================================
        # --------------------------------------------------------------------------------------------------------

    def read_bunch_of_images_from_dataset_tr(self, is_training):  # for training
        if settings.tr_isread == False:
            return
        settings.read_patche_mutex_tr.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)

        if len(self.random_images) < self.bunch_of_images_no:  # if there are no image choices for selection
            self.random_images = list(range(0, len(self.data)))
            self.deform1stdb += 1
        # select some distinct images for extracting patches!
        rand_image_no = np.random.randint(0, 10, self.bunch_of_images_no)

        self.random_images = [x for x in range(len(self.random_images)) if
                              x not in rand_image_no]  # remove selected images from the choice list
        print(rand_image_no)

        for img_index in range(len(rand_image_no)):
            deform = 0
            max_dis = 0
            imm = self.read_image(rand_image_no[img_index], deform=deform, max_dis=max_dis,tr=1)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('train image no read so far: %s' % len(self.collection))
        if is_training == True:
            settings.tr_isread = False
        else:
            settings.vl_isread = False
        settings.read_patche_mutex_tr.release()
        #==============================================================
    def read_bunch_of_images_from_dataset_mixedup_tr(self, is_training):  # for training
        if settings.tr_isread == False:
            return
        settings.read_patche_mutex_tr.acquire()
        self.collection.clear()
        self.seed += 1
        np.random.seed(self.seed)
        mixedup_no=int(self.bunch_of_images_no*.25)
        if len(self.random_images) < self.bunch_of_images_no+mixedup_no:  # if there are no image choices for selection
            self.random_images = list(range(0, len(self.data)))
            self.deform1stdb += 1
        # select some distinct images for extracting patches!
        rand_image_no = np.random.randint(0, 10, self.bunch_of_images_no+mixedup_no)

        self.random_images = [x for x in range(len(self.random_images)) if
                              x not in rand_image_no]  # remove selected images from the choice list
        print(rand_image_no)

        for img_index in range(len(rand_image_no)):
            deform = 0
            max_dis = 0
            imm = self.read_image(rand_image_no[img_index], deform=deform, max_dis=max_dis,tr=1)
            if len(imm) == 0:
                continue

            self.collection.append(imm)
            print('train image no read so far: %s' % len(self.collection))

        rand_mixedup = np.random.randint(0, self.bunch_of_images_no, mixedup_no)
        for i in range(len(rand_mixedup)):
            t=np.random.beta(.4,.4)
            for j in range(len(self.collection[i].crush_encoded)):
                self.collection[i].crush_encoded[j] = t * self.collection[i].crush_encoded[j] + (1-t)*self.collection[self.bunch_of_images_no+i].crush_encoded[j]
                self.collection[i].noncrush_encoded[j] = t * self.collection[i].noncrush_encoded[j] +(1-t)*self.collection[self.bunch_of_images_no+i].noncrush_encoded[j]
                if j< len(self.collection[i].angio):
                    if 1-t>t:
                        self.collection[i].angio[j] = self.collection[self.bunch_of_images_no+i].angio[j]
                        self.collection[i].perfusion[j] = self.collection[self.bunch_of_images_no+i].perfusion[j]
                        self.collection[i].mri[j] = self.collection[self.bunch_of_images_no+i].mri[j]
                        self.collection[i].segmentation[j] = self.collection[self.bunch_of_images_no+i].segmentation[j]
        self.collection=self.collection[0:self.bunch_of_images_no]

        if is_training == True:
            settings.tr_isread = False
        else:
            settings.vl_isread = False
        settings.read_patche_mutex_tr.release()



    # --------------------------------------------------------------------------------------------------------


        # shuffling the patches
    def shuffle_lists(self, crush_encoded_patches,noncrush_encoded_patches,perfusion_patches,angio_patches,mri_patches,segmentation_patches):
        index_shuf = list(range(np.shape(crush_encoded_patches)[1]))
        shuffle(index_shuf)
        crush_encoded_patches2 = np.vstack([crush_encoded_patches[:,sn,:,:,:]]
                                     for sn in index_shuf)
        noncrush_encoded_patches2 = np.vstack([noncrush_encoded_patches[:,sn,:,:,:]]
                                           for sn in index_shuf)
        perfusion_patches2 = np.vstack([perfusion_patches[:,sn,:,:,:]]
                                           for sn in index_shuf)
        angio_patches2 = np.vstack([angio_patches[:,sn,:,:,:]]
                                           for sn in index_shuf)
        mri_patches2 = np.vstack([mri_patches[:, sn, :, :, :]]
                                   for sn in index_shuf)
        segmentation_patches2 = np.vstack([segmentation_patches[:, sn, :, :, :]]
                                 for sn in index_shuf)
        return crush_encoded_patches2,noncrush_encoded_patches2,perfusion_patches2,angio_patches2,mri_patches2,segmentation_patches2
    #--------------------------------------------------------------------------------------------------------
    def extract_patch_from_group(self,group,rand_depth,rand_width,rand_height,patch_size):
        # patch_group=[]
        # for i in range(len(group)):
        #
        #     for sn in  range(len(rand_height)):
        #
        #         if (np.shape(group[i][
        #            int(rand_depth[sn]) - int(self.patch_window / 2) - 1:
        #            int(rand_depth[sn]) + int(self.patch_window / 2),
        #            int(rand_width[sn]) - int(self.patch_window / 2) - 1:
        #            int(rand_width[sn]) + int(self.patch_window / 2)
        #            , int(rand_height[sn]) - int(self.patch_window / 2) - 1:
        #            int(rand_height[sn]) + int(self.patch_window / 2)
        #            ])
        #           )!=(patch_size,patch_size,patch_size):
        #             print(rand_depth[sn],rand_height[sn],rand_width[sn])
        #             # plt.imshow(group[0][sn, :, :])
        #             # plt.show()
        #
        #     patches=np.stack([(group[i][
        #                int(rand_depth[sn]) - int(patch_size / 2) - 1:
        #                int(rand_depth[sn]) + int(patch_size / 2),
        #                int(rand_width[sn]) - int(patch_size / 2) - 1:
        #                int(rand_width[sn]) + int(patch_size / 2)
        #                , int(rand_height[sn]) - int(patch_size / 2) - 1:
        #                int(rand_height[sn]) + int(patch_size / 2)
        #                ]).astype(np.float32)
        #               for sn in
        #               range(len(rand_height))])
        #     patch_group.append(patches)
        patch_group = np.stack([[(group[i][
                              int(rand_depth[sn]) - int(patch_size / 2) - 1:
                              int(rand_depth[sn]) + int(patch_size / 2),
                              int(rand_width[sn]) - int(patch_size / 2) - 1:
                              int(rand_width[sn]) + int(patch_size / 2)
                              , int(rand_height[sn]) - int(patch_size / 2) - 1:
                              int(rand_height[sn]) + int(patch_size / 2)
                              ]).astype(np.float32)
                             for sn in
                             range(len(rand_height))] for i in range(len(group))])

        return patch_group
    #--------------------------------------------------------------------------------------------------------
            # read patches from the images which are in the RAM

    def read_patche_online_from_image_bunch_vl(self):
        if settings.vl_isread == True:
            return

        if len(self.collection) < self.bunch_of_images_no:
            return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_vl.acquire()
        print('start reading:%d' % len(self.collection))
        patch_no_per_image = int(self.sample_no_per_bunch / len(self.collection))
        # if patch_no_per_image==0:
        #     patch_no_per_image=1

        while patch_no_per_image * len(self.collection) < self.sample_no_per_bunch:
            patch_no_per_image += 1
        image_patches = []
        crush_encoded_patches2=[]
        noncrush_encoded_patches2=[]
        perfusion_patches2=[]
        angio_patches2=[]
        mri_patches2=[]
        segmentation_patches2=[]

        for ii in range(len(self.collection)):
            crush_encoded=self.collection[ii].crush_encoded
            noncrush_encoded=self.collection[ii].noncrush_encoded
            perfusion=self.collection[ii].perfusion
            angio=self.collection[ii].angio
            mri = [self.collection[ii].mri]
            segmentation=[self.collection[ii].segmentation]

            name=[self.collection[ii].name]


            # direction=self.collection[ii].direction
            # origin=self.collection[ii].origin
            # spacing=self.collection[ii].spacing

            #======================================
            # print('crush_encoded: %d, noncrush_encoded: %d, perfusion: %d, angio: %d, mri: %d, segmentation: %d'%(len(crush_encoded),len(noncrush_encoded),len(perfusion),len(angio),len(mri),len(segmentation)))
            random_range = []

            for i in range(len(angio)):

                if len(random_range):
                    random_range = np.hstack((random_range, np.where(angio[i])))
                else:
                    random_range = np.where(angio[i])
                # print('angio %d, %d'%(i,len(angio[i])))

            # print(' len(random_range[0]):%d, %s'%(len(random_range[0]),name))

            random1=np.random.randint(1, len(random_range[0]),
                                             size=int(np.ceil(
                                                 patch_no_per_image*.75)))  # get half depth samples from big arteries
            '''random numbers for selecting random samples'''

            extraction_range = np.where(crush_encoded[0] != 0)
            random2 = np.random.randint(1, len(extraction_range[0]),
                                             size=int(np.ceil(
                                                 patch_no_per_image*.25)))  # get half depth samples from every where



            rand_depth1 = random_range[0][random1]
            rand_width1 = random_range[1][random1]
            rand_height1 = random_range[2][random1]

            rand_depth2 = extraction_range[0][random2]
            rand_width2 = extraction_range[1][random2]
            rand_height2 = extraction_range[2][random2]

            rand_depth=np.hstack((rand_depth1,rand_depth2))
            rand_width=np.hstack((rand_width1,rand_width2))
            rand_height=np.hstack((rand_height1,rand_height2))

            crush_encoded_patches=self.extract_patch_from_group( crush_encoded, rand_depth, rand_width, rand_height,self.patch_window)
            noncrush_encoded_patches=self.extract_patch_from_group( noncrush_encoded, rand_depth, rand_width, rand_height,self.patch_window)
            perfusion_patches=self.extract_patch_from_group( perfusion, rand_depth, rand_width, rand_height, self.label_patch_size)
            angio_patches=self.extract_patch_from_group( angio, rand_depth, rand_width, rand_height, self.label_patch_size)
            mri_patches=self.extract_patch_from_group( mri, rand_depth, rand_width, rand_height, self.patch_window)
            segmentation_patches=self.extract_patch_from_group( segmentation, rand_depth, rand_width, rand_height, self.label_patch_size)

            # if np.shape(crush_encoded_patches)[0] != np.shape(noncrush_encoded_patches)[0] or np.shape(
            #         crush_encoded_patches)[0] != np.shape(perfusion_patches)[0] or np.shape(
            #         crush_encoded_patches)[0] != np.shape(angio_patches)[0] or np.shape(noncrush_encoded_patches)[0] != np.shape(
            #         angio_patches)[0] or np.shape(noncrush_encoded_patches)[0] != np.shape(perfusion_patches)[0] or np.shape(
            #         noncrush_encoded_patches)[0] != np.shape(angio_patches)[0]:
            #     print('Error: sizes are different, continue (vl)! ')
            #     continue

            if len(segmentation_patches2):
                crush_encoded_patches2=np.concatenate([crush_encoded_patches2,crush_encoded_patches],axis=1)
                noncrush_encoded_patches2=np.concatenate([noncrush_encoded_patches2,noncrush_encoded_patches],axis=1)
                perfusion_patches2=np.concatenate([perfusion_patches2,perfusion_patches],axis=1)
                angio_patches2=np.concatenate([angio_patches2,angio_patches],axis=1)
                mri_patches2=np.concatenate([mri_patches2,mri_patches],axis=1)
                segmentation_patches2=np.concatenate([segmentation_patches2,segmentation_patches],axis=1)

            else:
                crush_encoded_patches2=crush_encoded_patches
                noncrush_encoded_patches2=noncrush_encoded_patches
                perfusion_patches2=perfusion_patches
                angio_patches2=angio_patches
                mri_patches2=mri_patches
                segmentation_patches2=segmentation_patches

        if len(segmentation_patches2):

            [crush_encoded_patches2, noncrush_encoded_patches2,
             perfusion_patches2, angio_patches2,
             mri_patches2,segmentation_patches2] = self.shuffle_lists(crush_encoded_patches2, noncrush_encoded_patches2,
                                                                      perfusion_patches2, angio_patches2,mri_patches2,segmentation_patches2)


            if len(settings.subjects_vl2_segmentation) == 0:
                settings.subjects_vl2_crush = crush_encoded_patches2
                settings.subjects_vl2_noncrush = noncrush_encoded_patches2
                settings.subjects_vl2_perf = perfusion_patches2
                settings.subjects_vl2_angio = angio_patches2
                settings.subjects_vl2_mri = mri_patches2
                settings.subjects_vl2_segmentation = segmentation_patches2
            else:
                settings.subjects_vl2_crush = np.vstack((settings.subjects_vl2_crush, crush_encoded_patches2))
                settings.subjects_vl2_noncrush = np.vstack((settings.subjects_vl2_noncrush, noncrush_encoded_patches2))
                settings.subjects_vl2_perf = np.vstack((settings.subjects_vl2_perf, perfusion_patches2))
                settings.subjects_vl2_angio = np.vstack((settings.subjects_vl2_angio, angio_patches2))
                settings.subjects_vl2_mri = np.vstack((settings.subjects_vl2_mri, mri_patches2))
                settings.subjects_vl2_segmentation = np.vstack((settings.subjects_vl2_segmentation, segmentation_patches2))


        settings.vl_isread=True
        settings.read_patche_mutex_vl.release()



    #--------------------------------------------------------------------------------------------------------
    #read patches from the images which are in the RAM
    def read_patche_online_from_image_bunch_tr(self):

        if settings.tr_isread == True:
            return

        if len(self.collection) < self.bunch_of_images_no:
            return
        if  len(settings.subjects_tr_segmentation):
            if (np.shape(settings.subjects_tr_segmentation)[0])>200:
                return
        self.seed += 1
        np.random.seed(self.seed)
        settings.read_patche_mutex_tr.acquire()
        print('start reading:%d' % len(self.collection))
        patch_no_per_image = int(self.sample_no_per_bunch / len(self.collection))
        # if patch_no_per_image==0:
        #     patch_no_per_image=1

        while patch_no_per_image * len(self.collection) < self.sample_no_per_bunch:
            patch_no_per_image += 1
        crush_encoded_patches2 = []
        noncrush_encoded_patches2 = []
        perfusion_patches2 = []
        angio_patches2 = []
        mri_patches2 = []
        segmentation_patches2 = []
        for ii in range(len(self.collection)):
            crush_encoded=self.collection[ii].crush_encoded
            noncrush_encoded=self.collection[ii].noncrush_encoded
            perfusion=self.collection[ii].perfusion
            angio=self.collection[ii].angio
            mri=[self.collection[ii].mri]
            segmentation=[self.collection[ii].segmentation]

            direction=self.collection[ii].direction
            origin=self.collection[ii].origin
            spacing=self.collection[ii].spacing

            '''random numbers for selecting random samples'''
            # ======================================

            random_range = []
            for i in range(len(angio)):
                if len(random_range):
                    random_range = np.hstack((random_range, np.where(angio[i])))
                else:
                    random_range = np.where(angio[i])

            random1 = np.random.randint(1, len(random_range[0]),
                                        size=int(np.ceil(
                                            patch_no_per_image * .75)))  # get half depth samples from big arteries

            extraction_range = np.where(crush_encoded[0] != 0)
            random2 = np.random.randint(1, len(extraction_range[0]),
                                        size=int(np.ceil(
                                            patch_no_per_image * .25)))  # get half depth samples from every where

            rand_depth1 = random_range[0][random1]
            rand_width1 = random_range[1][random1]
            rand_height1 = random_range[2][random1]

            rand_depth2 = extraction_range[0][random2]
            rand_width2 = extraction_range[1][random2]
            rand_height2 = extraction_range[2][random2]

            rand_depth = np.hstack((rand_depth1, rand_depth2))
            rand_width = np.hstack((rand_width1, rand_width2))
            rand_height = np.hstack((rand_height1, rand_height2))



            crush_encoded_patches=self.extract_patch_from_group( crush_encoded, rand_depth, rand_width, rand_height,self.patch_window)
            noncrush_encoded_patches=self.extract_patch_from_group( noncrush_encoded, rand_depth, rand_width, rand_height,self.patch_window)
            perfusion_patches=self.extract_patch_from_group( perfusion, rand_depth, rand_width, rand_height, self.label_patch_size)
            angio_patches=self.extract_patch_from_group( angio, rand_depth, rand_width, rand_height, self.label_patch_size)
            mri_patches=self.extract_patch_from_group( mri, rand_depth, rand_width, rand_height, self.patch_window)
            segmentation_patches=self.extract_patch_from_group( segmentation, rand_depth, rand_width, rand_height, self.label_patch_size)
            # if np.shape(crush_encoded_patches)[0] != np.shape(noncrush_encoded_patches)[0] or np.shape(
            #         crush_encoded_patches)[0] != np.shape(perfusion_patches)[0] or np.shape(
            #         crush_encoded_patches)[0] != np.shape(angio_patches)[0] or np.shape(noncrush_encoded_patches)[0] != np.shape(
            #         angio_patches)[0] or np.shape(noncrush_encoded_patches)[0] != np.shape(perfusion_patches)[0] or np.shape(
            #         noncrush_encoded_patches)[0] != np.shape(angio_patches)[0]:
            #     print('Error: sizes are different, continue (tr)! ')
            #     continue

            if len(crush_encoded_patches2):
                crush_encoded_patches2=np.concatenate([crush_encoded_patches2,crush_encoded_patches],axis=1)
                noncrush_encoded_patches2=np.concatenate([noncrush_encoded_patches2,noncrush_encoded_patches],axis=1)
                perfusion_patches2=np.concatenate([perfusion_patches2,perfusion_patches],axis=1)
                angio_patches2=np.concatenate([angio_patches2,angio_patches],axis=1)
                mri_patches2=np.concatenate([mri_patches2,mri_patches],axis=1)
                segmentation_patches2=np.concatenate([segmentation_patches2,segmentation_patches],axis=1)

            else:
                crush_encoded_patches2=crush_encoded_patches
                noncrush_encoded_patches2=noncrush_encoded_patches
                perfusion_patches2=perfusion_patches
                angio_patches2=angio_patches
                mri_patches2=mri_patches
                segmentation_patches2=segmentation_patches

        if len(crush_encoded_patches2):
            [crush_encoded_patches2, noncrush_encoded_patches2,
             perfusion_patches2, angio_patches2,mri_patches2,segmentation_patches2] = self.shuffle_lists(crush_encoded_patches2, noncrush_encoded_patches2,
                                                                      perfusion_patches2, angio_patches2,mri_patches2,segmentation_patches2)



            if len(settings.subjects_tr2_angio) == 0:
                settings.subjects_tr2_crush = crush_encoded_patches2
                settings.subjects_tr2_noncrush = noncrush_encoded_patches2
                settings.subjects_tr2_perf = perfusion_patches2
                settings.subjects_tr2_angio = angio_patches2
                settings.subjects_tr2_mri = mri_patches2
                settings.subjects_tr2_segmentation = segmentation_patches2
            else:
                settings.subjects_tr2_crush = np.vstack((settings.subjects_tr2_crush, crush_encoded_patches2))
                settings.subjects_tr2_noncrush = np.vstack((settings.subjects_tr2_noncrush, noncrush_encoded_patches2))
                settings.subjects_tr2_perf = np.vstack((settings.subjects_tr2_perf, perfusion_patches2))
                settings.subjects_tr2_angio = np.vstack((settings.subjects_tr2_angio, angio_patches2))
                settings.subjects_tr2_mri = np.vstack((settings.subjects_tr2_mri, mri_patches2))
                settings.subjects_tr2_segmentation = np.vstack((settings.subjects_tr2_segmentation, segmentation_patches2))


        settings.tr_isread=True
        settings.read_patche_mutex_tr.release()

    # --------------------------------------------------------------------------------------------------------
    def return_patches_tr(self, batch_no):
        crush = []
        noncrsuh = []
        perf = []
        angio = []
        mri = []
        segmentation = []
        settings.train_queue.acquire()
        if len(settings.subjects_tr_segmentation):
            if np.shape(settings.subjects_tr_segmentation)[0] >= batch_no:
                angio = settings.subjects_tr_angio[0:batch_no, :, :, :, :]
                perf = settings.subjects_tr_perf[0:batch_no, :, :, :, :]
                crush = settings.subjects_tr_crush[0:batch_no, :, :, :, :]
                noncrsuh = settings.subjects_tr_noncrush[0:batch_no, :, :, :, :]
                mri = settings.subjects_tr_mri[0:batch_no, :, :, :, :]
                segmentation = settings.subjects_tr_segmentation[0:batch_no, :, :, :, :]

                settings.subjects_tr_angio =np.delete(settings.subjects_tr_angio, range(batch_no), axis=0)
                settings.subjects_tr_perf =np.delete(settings.subjects_tr_perf, range(batch_no), axis=0)
                settings.subjects_tr_crush =np.delete(settings.subjects_tr_crush, range(batch_no), axis=0)
                settings.subjects_tr_noncrush =np.delete(settings.subjects_tr_noncrush, range(batch_no), axis=0)
                settings.subjects_tr_mri =np.delete(settings.subjects_tr_mri, range(batch_no), axis=0)
                settings.subjects_tr_segmentation =np.delete(settings.subjects_tr_segmentation, range(batch_no), axis=0)

            else:
                settings.subjects_tr_angio = np.delete(settings.subjects_tr_angio, range(len(
                    settings.subjects_tr_angio)), axis=0)
                settings.subjects_tr_perf = np.delete(settings.subjects_tr_perf,
                                                      range(len(settings.subjects_tr_perf)), axis=0)
                settings.subjects_tr_crush = np.delete(settings.subjects_tr_crush, range(len(
                    settings.subjects_tr_crush)), axis=0)
                settings.subjects_tr_noncrush = np.delete(settings.subjects_tr_noncrush, range(len(
                    settings.subjects_tr_noncrush)), axis=0)
                settings.subjects_tr_mri = np.delete(settings.subjects_tr_mri , range(len(
                    settings.subjects_tr_mri)), axis=0)
                settings.subjects_tr_segmentation = np.delete(settings.subjects_tr_segmentation, range(len(
                    settings.subjects_tr_segmentation)), axis=0)

            if np.shape(noncrsuh) != np.shape(crush) \
                    or np.shape(perf) != np.shape(angio) \
                    or np.shape(segmentation)[0] != np.shape(perf)[0]\
                    or np.shape(segmentation)[0] != np.shape(angio)[0] \
                    or np.shape(segmentation)[0] != np.shape(mri)[0]\
                    or np.shape(crush)[0] != np.shape(noncrsuh)[0]\
                    or np.shape(angio)[0] != np.shape(mri)[0]:
                print("somthing wrong with size tr!!")
                crush = []
                noncrsuh = []
                perf = []
                angio = []
                mri = []
                segmentation = []
        settings.train_queue.release()

        return np.expand_dims(crush, axis=5), \
               np.expand_dims(noncrsuh, axis=5), \
               np.expand_dims(perf, axis=5), \
               np.expand_dims(angio, axis=5),\
               np.expand_dims(mri, axis=5),\
               np.expand_dims(segmentation, axis=5), \
    # --------------------------------------------------------------------------------------------------------

    def return_patches_mixedup_tr(self, batch_no):
        crush = []
        noncrsuh = []
        perf = []
        angio = []
        mri = []
        segmentation = []
        settings.train_queue.acquire()
        mixedup_no=int(batch_no*.75)
        if len(settings.subjects_tr_segmentation):
            # if np.shape(settings.subjects_tr_segmentation)[0] >= batch_no+mixedup_no:
            #     angio = settings.subjects_tr_angio[0:batch_no+mixedup_no, :, :, :, :]
            #     perf = settings.subjects_tr_perf[0:batch_no+mixedup_no, :, :, :, :]
            #     crush = settings.subjects_tr_crush[0:batch_no+mixedup_no, :, :, :, :]
            #     noncrsuh = settings.subjects_tr_noncrush[0:batch_no+mixedup_no, :, :, :, :]
            #     mri = settings.subjects_tr_mri[0:batch_no+mixedup_no, :, :, :, :]
            #     segmentation = settings.subjects_tr_segmentation[0:batch_no+mixedup_no, :, :, :, :]
            #
            #     rand_mixedup = np.random.randint(0, batch_no, mixedup_no)
            #     for i in range(len(rand_mixedup)):
            #         t=np.random.beta(.4,.4)
            #         crush[rand_mixedup[i], :, :, :, :] =t*crush[rand_mixedup[i], :, :, :, :] +(1-t)*crush[batch_no+i, :, :, :, :]
            #         noncrsuh[rand_mixedup[i], :, :, :, :] =t*noncrsuh[rand_mixedup[i], :, :, :, :]+(1-t)*noncrsuh[batch_no+i, :, :, :, :]
            #         if 1-t>t:
            #             angio[rand_mixedup[i], :, :, :, :] = angio[batch_no+i, :, :, :, :]
            #             perf[rand_mixedup[i], :, :, :, :] = perf[batch_no+i, :, :, :, :]
            #             mri[rand_mixedup[i], :, :, :, :] = mri[batch_no+i, :, :, :, :]
            #             segmentation[rand_mixedup[i], :, :, :, :] = segmentation[batch_no+i, :, :, :, :]
            #
            #     angio = angio[0:batch_no , :, :, :, :]
            #     perf = perf[0:batch_no , :, :, :, :]
            #     crush = crush[0:batch_no , :, :, :, :]
            #     noncrsuh = noncrsuh[0:batch_no , :, :, :, :]
            #     mri = mri[0:batch_no , :, :, :, :]
            #     segmentation = segmentation[0:batch_no , :, :, :, :]


            if np.shape(settings.subjects_tr_segmentation)[0] >= batch_no:
                angio = settings.subjects_tr_angio[0:batch_no, :, :, :, :]
                perf = settings.subjects_tr_perf[0:batch_no, :, :, :, :]
                crush = settings.subjects_tr_crush[0:batch_no, :, :, :, :]
                noncrsuh = settings.subjects_tr_noncrush[0:batch_no, :, :, :, :]
                mri = settings.subjects_tr_mri[0:batch_no, :, :, :, :]
                segmentation = settings.subjects_tr_segmentation[0:batch_no, :, :, :, :]




            else:
                settings.subjects_tr_angio = np.delete(settings.subjects_tr_angio, range(len(
                    settings.subjects_tr_angio)), axis=0)
                settings.subjects_tr_perf = np.delete(settings.subjects_tr_perf,
                                                      range(len(settings.subjects_tr_perf)), axis=0)
                settings.subjects_tr_crush = np.delete(settings.subjects_tr_crush, range(len(
                    settings.subjects_tr_crush)), axis=0)
                settings.subjects_tr_noncrush = np.delete(settings.subjects_tr_noncrush, range(len(
                    settings.subjects_tr_noncrush)), axis=0)
                settings.subjects_tr_mri = np.delete(settings.subjects_tr_mri, range(len(
                    settings.subjects_tr_mri)), axis=0)
                settings.subjects_tr_segmentation = np.delete(settings.subjects_tr_segmentation, range(len(
                    settings.subjects_tr_segmentation)), axis=0)

            if np.shape(noncrsuh) != np.shape(crush) \
                    or np.shape(perf) != np.shape(angio) \
                    or np.shape(segmentation)[0] != np.shape(perf)[0] \
                    or np.shape(segmentation)[0] != np.shape(angio)[0] \
                    or np.shape(segmentation)[0] != np.shape(mri)[0] \
                    or np.shape(crush)[0] != np.shape(noncrsuh)[0] \
                    or np.shape(angio)[0] != np.shape(mri)[0]:
                print("somthing wrong with size tr!!")
                crush = []
                noncrsuh = []
                perf = []
                angio = []
                mri = []
                segmentation = []
        settings.train_queue.release()

        return np.expand_dims(crush, axis=5), \
               np.expand_dims(noncrsuh, axis=5), \
               np.expand_dims(perf, axis=5), \
               np.expand_dims(angio, axis=5), \
               np.expand_dims(mri, axis=5), \
               np.expand_dims(segmentation, axis=5), \
    #--------------------------------------------------------------------------------------------------------
    def return_patches_tr_seg(self,batch_no):
        crush = []
        noncrsuh = []
        perf = []
        angio = []
        seg_label=[]
        mri=[]
        segmentation=[]
        settings.train_queue.acquire()
        if len(settings.subjects_tr_segmentation):
            if np.shape(settings.subjects_tr_segmentation)[0] >= batch_no:
                angio = settings.subjects_tr_angio[0:batch_no, :, :, :, :]
                perf = settings.subjects_tr_perf[0:batch_no, :, :, :, :]
                crush = settings.subjects_tr_crush[0:batch_no, :, :, :, :]
                noncrsuh = settings.subjects_tr_noncrush[0:batch_no, :, :, :, :]
                mri = settings.subjects_tr_mri[0:batch_no, :, :, :, :]
                segmentation = settings.subjects_tr_segmentation[0:batch_no, :, :, :, :]

            else:
                settings.subjects_tr_angio = np.delete(settings.subjects_tr_angio, range(len(
                    settings.subjects_tr_angio)), axis=0)
                settings.subjects_tr_perf = np.delete(settings.subjects_tr_perf, range(len(settings.subjects_tr_perf)), axis=0)
                settings.subjects_tr_crush = np.delete(settings.subjects_tr_crush, range(len(
                    settings.subjects_tr_crush)), axis=0)
                settings.subjects_tr_noncrush = np.delete(settings.subjects_tr_noncrush, range(len(
                    settings.subjects_tr_noncrush)), axis=0)

                settings.subjects_tr_mri = np.delete(settings.subjects_tr_mri, range(len(
                    settings.subjects_tr_mri)), axis=0)
                settings.subjects_tr_segmentation = np.delete(settings.subjects_tr_segmentation, range(len(
                    settings.subjects_tr_segmentation)), axis=0)

            seg_label = np.maximum(angio, 0)
            seg_label = np.minimum(1000000 * seg_label, 1)

        settings.train_queue.release()



        return np.expand_dims(crush, axis=5), \
               np.expand_dims(noncrsuh, axis=5), \
               np.expand_dims(perf, axis=5), \
               np.expand_dims(angio, axis=5),\
               np.expand_dims(seg_label, axis=5),\
               np.expand_dims(mri, axis=5),\
               np.expand_dims(segmentation, axis=5),\



    #--------------------------------------------------------------------------------------------------------
    def return_patches_vl(self, start,end,is_tr):
        crush=[]
        noncrsuh=[]
        perf=[]
        angio=[]
        mri=[]
        segmentation=[]
        if len(settings.subjects_vl_segmentation):
            if (np.shape(settings.subjects_vl_segmentation)[0]  - (end)) >= 0 :
                angio = settings.subjects_vl_angio[start:end, :, :, :, :]
                perf = settings.subjects_vl_perf[start:end, :, :, :, :]
                crush = settings.subjects_vl_crush[start:end, :, :, :, :]
                noncrsuh = settings.subjects_vl_noncrush[start:end, :, :, :, :]
                mri = settings.subjects_vl_mri[start:end, :, :, :, :]
                segmentation = settings.subjects_vl_segmentation[start:end, :, :, :, :]
            if np.shape(angio)!=np.shape(perf) or np.shape(crush)!=np.shape(noncrsuh) or \
                    np.shape(segmentation)[0]!=np.shape(noncrsuh)[0] \
                    or np.shape(segmentation)[0]!=np.shape(perf)[0] \
                    or np.shape(segmentation)[0]!=np.shape(crush)[0] \
                    or np.shape(segmentation)[0]!=np.shape(mri)[0] \
                    or np.shape(segmentation)[0] != np.shape(angio)[0]:
                print("somthing wrong with size vl!!")
                crush = []
                noncrsuh = []
                perf = []
                angio = []
                mri =[]
                segmentation =[]


        return np.expand_dims(crush, axis=5),\
               np.expand_dims(noncrsuh, axis=5),\
               np.expand_dims(perf, axis=5),\
               np.expand_dims(angio, axis=5),\
               np.expand_dims(mri, axis=5),\
               np.expand_dims(segmentation, axis=5),

    # -------------------------------------------------------------------------------------------------------
    def return_patches_vl_seg(self, start, end, is_tr):
        crush = []
        noncrsuh = []
        perf = []
        angio = []
        seg_label=[]
        mri = []
        segmentation = []
        if len(settings.subjects_vl_angio):
            if (np.shape(settings.subjects_vl_segmentation)[0] - (end)) >= 0:
                angio = settings.subjects_vl_angio[start:end, :, :, :, :]
                perf = settings.subjects_vl_perf[start:end, :, :, :, :]
                crush = settings.subjects_vl_crush[start:end, :, :, :, :]
                noncrsuh = settings.subjects_vl_noncrush[start:end, :, :, :, :]
                mri = settings.subjects_vl_mri[start:end, :, :, :, :]
                segmentation = settings.subjects_vl_segmentation[start:end, :, :, :, :]

                seg_label = np.maximum(angio, 0)
                seg_label = np.minimum(1000000 * seg_label, 1)

        return np.expand_dims(crush, axis=5), \
               np.expand_dims(noncrsuh, axis=5), \
               np.expand_dims(perf, axis=5), \
               np.expand_dims(angio, axis=5),\
               np.expand_dims(seg_label, axis=5), \
               np.expand_dims(mri, axis=5), \
               np.expand_dims(segmentation, axis=5),
    # -------------------------------------------------------------------------------------------------------
