import functions.analysis as anly
import SimpleITK as sitk
if __name__=='__main__':
    path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/multi_stage/'
    experiment = 'experiment-2/'
    results = 'results/'
    subject='18_PP5_53_cinema2/'
    type = 'perf_0.mha'
    gt_pth= path+experiment+results+subject+'GT/'+type
    img_pth= path+experiment+results+subject+type

    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_pth))
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_pth))

    max_angio = 2.99771428108
    max_perf = 17.0151833445

    print(anly.analysis(img, gt, 0, max_perf))
