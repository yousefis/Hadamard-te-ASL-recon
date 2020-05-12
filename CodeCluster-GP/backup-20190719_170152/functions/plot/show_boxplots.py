import matplotlib.cm as cm
import functions.analysis as anly
from openpyxl import load_workbook
import xlsxwriter
import pandas
import pandas as pd

from shutil import copyfile
import os,sys,inspect
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from functions.image_reader.read_data import _read_data
from functions.image_reader.image_class import image_class
from functions.loss.loss_fun import _loss_func
from functions.network.densenet_angio_perf import _densenet
path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG'

excel_file = [path+'/normal_SSIM_p_a_newdb/experiment-1/results/results.xlsx',
              path+'/multi_stage/experiment-2/results/results.xlsx']
data_to_plot=[]
plot_array=[]
if __name__=='__main__':
    titles=['SSIM','PSNR','SNR','MSE']
    fig = plt.figure(1, )
    tt=1
    c = ['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'orchid']
    c=np.repeat(c,np.size(excel_file))
    ylim=[[.9,1],[20,40],[0,15],[0,2]]
    for t in titles:
        plot_array=[]
        for file in excel_file:
            workbook = load_workbook(file,read_only=False)
            for k in range(7):
                worksheet = workbook['perf'+str(k)]   # make sure this sheet exist e.g. create it before
                df = pd.read_excel(file, 'perf'+str(k)) #sheet 1 contains the data see above
                if np.size(plot_array)==0:
                    plot_array=np.transpose([df._series[t]])
                else:
                    plot_array=np.append(plot_array,np.transpose([df._series[t]]),axis=1)


        ax = fig.add_subplot(4,1,tt)

        ax.set_ylim(ylim[tt-1])
        tt = tt + 1
        # for j in range(7):
        # j=1
        bp = ax.boxplot(plot_array, notch=True, patch_artist=True,
                # boxprops=dict(facecolor=c, color=c),
                # capprops=dict(color=c),
                # whiskerprops=dict(color=c),
                # flierprops=dict(color=c, markeredgecolor=c),
                # medianprops=dict(color=c),
                        showmeans=True)

        # for flier in bp['fliers']:
        #      flier.set(marker='o', color=c, alpha=0.5)

        # Fill with colors
        # cmap = cm.ScalarMappable(cmap='rainbow')
        # test_mean = [np.mean(x) for x in plot_array]
        # for patch, color in zip(bp['boxes'], cmap.to_rgba(test_mean)):
        #     patch.set_facecolor(color)

        cm = plt.cm.get_cmap('rainbow')
        colors = ['red', 'blue', 'green', 'pink', 'yellow', 'cyan', 'orchid']

        for patch, color in zip(bp['boxes'], colors):
            print(color)
            patch.set_facecolor(color)
            patch.set(color=color, linewidth=2)
            # patch.set(bp[element], color=edge_color)
            for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
                plt.setp(bp[element], color=color)
        # adding horizontal grid lines

        ax.yaxis.grid(True)
    # plt.xticks([1, 2,], ['SSIM','Multi-level'])
    plt.show()