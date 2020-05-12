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
path='/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG'

excel_file = [path+'/Workshop_log/MSE/experiment-2/results/results.xlsx',
              path+'/normal_SSIM_p_a_newdb/experiment-1/results/results.xlsx',
              path+'/Log_perceptual/regularization/perceptual-0/results/results.xlsx',
              path+'/multi_stage/experiment-2/results/results.xlsx',
              ]
data_to_plot=[]
plot_array=[]
plt.rcParams.update({'font.size': 16})
# fontdicty = {'fontsize': 16,
#                  'weight': 'bold',
#                  'verticalalignment': 'baseline',
#                  'horizontalalignment': 'center'}
if __name__=='__main__':
    titles=['SSIM','PSNR','MSE']
    fig = plt.figure(1, )

    colors = ['red', 'blue', 'green','cyan']#, 'pink', 'yellow', 'cyan', 'orchid']
    ylim=[[.9,0.95],[.6,1.],
          [20,40],[0,15],
          [0,4],[0,25],
          [0,4],[0,10]]
    rank=7
    ang_perf=['perf','angio']
    xlabl=['Perfusion', 'Angiography']
    v=1
    for v,ap in zip([1,2],ang_perf):
        i = 0

        for t in titles:

            plot_array=[]
            for file in excel_file:
                workbook = load_workbook(file,read_only=False)
                for k in range(rank):
                    worksheet = workbook[ap+str(k)]   # make sure this sheet exist e.g. create it before
                    df = pd.read_excel(file, ap+str(k)) #sheet 1 contains the data see above
                    if len(plot_array)==0:
                        plot_array=[np.array(df._series[t]/rank)]
                    else:
                        if k%rank==0 :
                            plot_array.append(np.array(df._series[t]/rank))
                        else:
                            plot_array[len(plot_array)-1]+=np.array(df._series[t]/rank)



            if v==1:
                tt = i*2 + 1
            else:
                tt=(i+1)*2
            ax = fig.add_subplot(3, 2, tt)
            plt.sca(ax)
            plt.ylabel(t)
            if i==0:
                plt.title(xlabl[v-1])
            print(tt)
            ax.set_ylim(ylim[tt-1])


            i+=1
            # for j in range(7):
            # j=1
            plot_array=np.transpose(np.array(plot_array))
            # bp = ax.boxplot(plot_array, notch=True, patch_artist=True,
                    # boxprops=dict(facecolor=c, color=c),
                    # capprops=dict(color=c),
                    # whiskerprops=dict(color=c),
                    # flierprops=dict(color=c, markeredgecolor=c),
                    # medianprops=dict(color=c),
                    #         showmeans=True)
            # plt.boxplot(plot_array,  notch=True, patch_artist=True,
            #             boxprops=dict(facecolor=c, color=c),
            #             capprops=dict(color=c),
            #             whiskerprops=dict(color=c),
            #             flierprops=dict(color=c, markeredgecolor=c),
            #             medianprops=dict(color=c),
            #             )
            box = plt.boxplot(plot_array, patch_artist=True,)
            for patch, color in zip(box['boxes'], colors):
                print(color)
                patch.set_facecolor(color)
                ## change color and linewidth of the whiskers
                for whisker in box['whiskers']:
                    whisker.set(color=color, linewidth=1)
                ## change color and linewidth of the caps
                for cap in box['caps']:
                    cap.set(color=color, linewidth=1)

                ## change the style of fliers and their fill
                for flier in box['fliers']:
                    flier.set(marker='o', color=color, alpha=0.5)

            # for flier in bp['fliers']:
            #      flier.set(marker='o', color=c, alpha=0.5)

            # Fill with colors
            # cmap = cm.ScalarMappable(cmap='rainbow')
            # test_mean = [np.mean(x) for x in plot_array]
            # for patch, color in zip(bp['boxes'], cmap.to_rgba(test_mean)):
            #     patch.set_facecolor(color)

            cm = plt.cm.get_cmap('rainbow')

            # for patch, color in zip(bp['boxes'], colors):
            #     patch.set_facecolor(color)


            # for patch, color in zip(bp['boxes'], colors):
            #     print(color)
            #     patch.set_facecolor(color)
            #     patch.set(color=color, linewidth=2)
            #     # patch.set(bp[element], color=edge_color)
            #     for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            #         plt.setp(bp[element], color=color)
            # adding horizontal grid lines
            plt.xticks([1, 2, 3,4], ['MSE','SSIM','PL', 'ML-SSIM'])
            ax.yaxis.grid(True)

    plt.show()