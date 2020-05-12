import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import ExcelWriter
import turtle
from pandas import ExcelFile
if __name__=='__main__':
    fig = plt.figure(1, )
    colors = ['hotpink', 'salmon', 'lightgreen', '#0066FF']  # , 'pink', 'yellow', 'cyan', 'orchid']
    ylim = [[90, 100], [-5, 105],
            [0, 0.45], [0, 4],
            [25, 45], [0, 80],]
    plt.rcParams.update({'font.size': 16})
    # ylim = [[0, 1.], [-.05, 1.05],
    #         [0, 1], [0, 15],
    #         [0, 100], [0, 100], ]

    xlabl = ['Perfusion', 'Angiography']
    resultsdir="results/"
    ang_perf = ['perf', 'angio']
    parent_path= "/exports/lkeb-hpc/syousefi/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/"
    filenames = [parent_path + "Workshop_log/MSE/experiment-2/"+resultsdir+"results.xlsx",
     parent_path + "normal_SSIM_p_a_newdb/experiment-1/"+resultsdir+"results.xlsx",
    parent_path + "Log_perceptual/regularization/perceptual-0/"+resultsdir +"results.xlsx",
                 parent_path + "multi_stage/experiment-2/" + resultsdir + "results.xlsx",]
    titles = ['SSIM', 'MSE', 'PSNR']
    rank=7
    pp=[]
    for v, ap in zip([1, 2], ang_perf):
        i = 0
        if v==1:
            titles_prim=['a) SSIM%','c) MSE','e) pSNR ']
        else:
            titles_prim =['b) SSIM%','d) MSE','f) pSNR']

        for t, tprim in zip(titles, titles_prim):
            plot_array=[]
            for file in filenames:
                for k in range(rank):
                    df = pd.read_excel(file, sheet_name=ap+str(k))
                    if k==0:
                        df_array =df[t].to_numpy()
                    else:
                        df_array = np.concatenate((df_array,df[t].to_numpy()))
                if t is "SSIM":
                    print("SSIM mean: %.1f "%(np.mean(df_array*100)))
                    print("SSIM var: %.1f "%(np.sqrt(np.var(df_array*100))))
                    print("SSIM median:%.1f "%(np.median(df_array*100)))
                    print("================================")
                elif t is "MSE":
                    print("MSE mean: %.2f "%(np.mean(df_array)))
                    print("MSE var: %.2f "%(np.sqrt(np.var(df_array))))
                    print("MSE median: %.2f "%(np.median(df_array)))
                    print("================================")
                elif t is "PSNR":
                    print("PSNR mean: %.2f "%(np.mean(df_array)))
                    print("PSNR var: %.2f "%(np.sqrt(np.var(df_array))))
                    print("PSNR median: %.2f "%((np.median(df_array))))
                    print("================================")
                # if len(plot_array) == 0:
                if t is "SSIM":
                    plot_array.append(100*df_array)
                else:
                    plot_array.append( df_array)
                # else:
                #     plot_array[len(plot_array) - 1] += df_array


            if v==1:
                tt = i*2 + 1
            else:
                tt=(i+1)*2
            ax = fig.add_subplot(3, 2, tt)
            plt.sca(ax)
            plt.ylabel(tprim)
            if i==0:
                plt.title(xlabl[v-1])
            print(tt)
            ax.set_ylim(ylim[tt-1])


            i+=1
            # for j in range(7):
            # j=1
            plot_array=np.transpose(np.array(plot_array))
            meanpointprops = dict(marker='D', markeredgecolor='black',
                                  markerfacecolor='firebrick')
            box = plt.boxplot(plot_array, patch_artist=True,meanprops=meanpointprops, showmeans=True,whis=[5, 95])
            pp.append(box)
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


            plt.xticks([1, 2, 3, 4], ['', '', '', ''])
            ax.yaxis.grid(True)
            # plt.legend(handles=[box['boxes']])
            # plt.legend(box, ['MSE', 'SSIM', 'PL', 'ML-SSIM'])

        # Create legend
        # plt.xticks([1, 2, 3, 4], ['MSE', 'SSIM', 'PL', 'ML-SSIM'])



    legends = []
    for poly,i in   zip( pp,range(4)):
        legends.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i]))
    # don't try to understand the legend displacement thing here. Believe me. Don't.
    plt.figlegend(legends, ['MSE', 'SSIM', 'PL', 'ML-SSIM'], loc=7)

    plt.show()