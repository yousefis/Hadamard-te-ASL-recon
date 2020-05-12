import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
resultsdir="results/"
filename = "/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/Workshop_log/MSE/experiment-2/"+resultsdir+"results.xlsx"
# filename = "/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/normal_SSIM_p_a_newdb/experiment-1/"+resultsdir+"results.xlsx"
# filename = "/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/multi_stage/experiment-2/"+resultsdir +"results.xlsx"
# filename = "/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/Log_perceptual/regularization/perceptual-0/"+resultsdir +"results.xlsx"
for i in range(7):
    df = pd.read_excel(filename, sheet_name='perf'+str(i))
    if i==0:
        SNR =df["SNR"].to_numpy()
        PSNR =df["PSNR"].to_numpy()
        MSE =df["MSE"].to_numpy()
        SSIM =df["SSIM"].to_numpy()
    else:
        SNR = np.concatenate((SNR,df["SNR"].to_numpy()))
        PSNR = np.concatenate((PSNR,df["PSNR"].to_numpy()))
        MSE = np.concatenate((MSE,df["MSE"].to_numpy()))
        SSIM = np.concatenate((SSIM,df["SSIM"].to_numpy()))
    # print("SSIM mean: ", str(np.mean(df["SSIM"].to_numpy())))

        # print("PSNR mean: ", str(np.mean(PSNR)))
        # print("PSNR var: ", str(np.sqrt(np.var(PSNR))))
        # print("PSNR median: ", str((np.median(PSNR))))
        # print("================================")
print("SSIM mean: %.2f "%(np.mean(SSIM*100)))
print("SSIM var: %.2f "%(np.sqrt(np.var(SSIM*100))))
print("SSIM median:%.2f "%(np.median(SSIM*100)))
print("================================")

print("MSE mean: %.2f "%(np.mean(MSE)))
print("MSE var: %.2f "%(np.sqrt(np.var(MSE))))
print("MSE median: %.2f "%(np.median(MSE)))
print("================================")
print("SNR mean: %.2f "%(np.mean(SNR)))
print("SNR var: %.2f "%(np.sqrt(np.var(SNR))))
print("SNR median: %.2f "%(np.median(SNR)))

print("================================")
print("PSNR mean: %.2f "%(np.mean(PSNR)))
print("PSNR var: %.2f "%(np.sqrt(np.var(PSNR))))
print("PSNR median: %.2f "%((np.median(PSNR))))


