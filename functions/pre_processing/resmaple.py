import SimpleITK as sitk
def resampling( image, new_spacing):
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


if __name__=='__main__':
    path='/exports/lkeb-hpc/syousefi/Data/invivo_decomposed/1/00_PP0_0_cinema1/decoded_non_crush/'
    # type='crush'
    type='decode_'
    ext='.nii'
    spacing=[3,3,3]
    for i in range(1,16):
        I=sitk.ReadImage(path+type+str(i)+ext)
        sitk_image = resampling(I,spacing)
        sitk_image.SetDirection(direction=I.GetDirection())
        sitk_image.SetOrigin(origin=I.GetOrigin())
        sitk_image.SetSpacing(spacing=spacing)
        sitk.WriteImage(sitk_image,path+'resam/'+type+str(i)+ext)
        print(path+'00_PP0_0_cinema0/decoded_crush/resam/'+type+str(i)+ '.mha done!')