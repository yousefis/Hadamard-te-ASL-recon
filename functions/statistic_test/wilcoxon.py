from scipy.stats import wilcoxon
import numpy as np
from openpyxl import load_workbook
import pandas as pd

if __name__=='__main__':
    alpha = 0.05
    path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG'
    rank = 7
    excel_file = [path+'/Workshop_log/MSE/experiment-2/results/results.xlsx',
                  path+'/normal_SSIM_p_a_newdb/experiment-1/results/results.xlsx',
                  path+'/Log_perceptual/regularization/perceptual-0/results/results.xlsx',
                  path+'/multi_stage/experiment-2/results/results.xlsx',
                  ]
    measure="SSIM"
    # measure="MSE"
    # measure="SNR"
    # measure="PSNR"
    # statistic, pvalue = wilcoxon(x, y, zero_method='zsplit')
    print("measure: "+measure)
    ang_perf = ['perf', 'angio']

    for v, ap in zip([1, 2], ang_perf):
        SSIM_list=[]
        for file in excel_file:
            workbook = load_workbook(file, read_only=False)
            for k in range(rank):
                worksheet = workbook[ap + str(k)]  # make sure this sheet exist e.g. create it before
                df = pd.read_excel(file, ap + str(k))  # sheet 1 contains the data see above
                if k == 0:
                    SSIM = df[measure].to_numpy()
                else:
                    SSIM = np.concatenate((SSIM, df[measure].to_numpy()))
            SSIM_list.append(SSIM)
            # print(np.average(SSIM_list))

        df = pd.DataFrame({'MSE-net': SSIM_list[0],
                           'SSIM-net': SSIM_list[1],
                           'PL-net': SSIM_list[2],
                           'multi_stage-net': SSIM_list[3]
                           })
        writer = pd.ExcelWriter('/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/esmrmb2019/'+measure+'.xlsx')
        df.to_excel(writer, 'Sheet1')
        writer.save()
        for i in range(len(excel_file)-1):
            statistic, pvalue = wilcoxon(SSIM_list[i], SSIM_list[3], zero_method='zsplit')
            if pvalue > alpha:
                #same distribution not significant
                print("Not significant")
            else:
                print("Significant")

            print("pvalue: "+str(pvalue))
    print("==========")


    # print('pvalue={:.2f}'.format(pvalue))
    # if pvalue > alpha:
    #    # same distribution
    #    significance_list = np.append(significance_list, False)
    # else:
    #    significance_list = np.append(significance_list, True)