import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import functions.plot.color_constants as color_constants
import functions.settings.setting_utils as su


def loss_plot():
    """
    This script reads the summary file(s) and plot the defined losses.
    This script reads all the summary files in one folder. So there is no problem if the training was paused and resumed.


    input: current_experiment_list: list of experiments to plot
           stage_list: list of stages to plot
           train_mode_list: ['Training' , 'Validation']
           average_length: the lenght of the window for moving average.

    Algorithm: It loads the summary file. By tf.train.summary_iterator checks all the steps in the summary file.
               Then add desired scalars: loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'Loss'].append(e.summary.value[8].simple_value)
               you need to check that your desired loss is stored in which element of the e.summary.value

    :return:
    """

    current_experiment_list = ['20181208_001201_sh', '20181208_100333_sh']
    stage_list = [1]
    train_mode_list = ['Validation']  # 'Training' , 'Validation'
    average_length = 20

    title_dict = {'2020_multistage_crop3': 'Max-Pooling',
                  '2020_multistage_dec3': 'Decimation',
                  '20180921_max15_D12_stage2_crop3': 'Learnable transposed convolution',
                  '20181109_max15_D12_stage2_crop4_old': 'Nearest neighbor interpolation',
                  '20181116_max15_D12_stage2_crop4': 'Trilinear interpolation'}
    for exp in current_experiment_list:
        if exp not in title_dict.keys():
            title_dict[exp] = exp
    loss_list = ['Huber', 'BE', 'Loss']
    loss_dict = dict()
    for current_experiment in current_experiment_list:
        setting = su.initialize_setting(current_experiment)
        if len(stage_list) > 1:
            setting, network_dict = get_backup_folder(setting, stage_list)
        else:
            network_dict = {'Stage'+str(stage_list[0]): {'NetworkLoad': current_experiment}}

        loss_dict[current_experiment] = dict()
        log_name_train_type = {'Training': 'train/',
                               'Validation': 'test/'}

        for stage in stage_list:
            network_exp = network_dict['Stage'+str(stage)]['NetworkLoad']
            loss_dict[current_experiment]['Stage'+str(stage)] = \
                {'TrainingStep': [], 'TrainingLoss': [], 'TrainingHuber': [], 'TrainingBE': [],
                 'ValidationStep': [], 'ValidationLoss': [], 'ValidationHuber': [], 'ValidationBE': []}

            log_folder = su.address_generator(setting, 'log_folder', current_experiment=network_exp)
            for train_mode in train_mode_list:
                log_test_folder = log_folder + log_name_train_type[train_mode]
                file_list = [f for f in os.listdir(log_test_folder) if os.path.isfile(os.path.join(log_test_folder, f))]
                file_list_time = [os.path.getmtime(log_test_folder + file) for file in file_list]
                file_list_time, file_list = (list(t) for t in zip(*sorted(zip(file_list_time, file_list))))

                for file in file_list:
                    print('Stage='+str(stage)+' load file'+file)
                    for e in tf.train.summary_iterator(log_test_folder + file):
                        if len(e.summary.value) > 0:
                            loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'Step'].append(e.step)
                            loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'Loss'].append(e.summary.value[8].simple_value)
                            loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'Huber'].append(e.summary.value[9].simple_value)
                            loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'BE'].append(e.summary.value[10].simple_value)
                            # print('step = {}'.format(e.step) + ' tag = ' + e.summary.value[9].tag + ' value = {:.3f} walltime = {}'.
                            #       format(e.summary.value[9].simple_value, time.asctime(time.localtime(e.wall_time))))

                    print('Stage=' + str(stage) + ' end file' + file)

        train_mode = 'Validation'
        stage = 2
        loss = 'Huber'
        plt.plot(loss_dict[current_experiment]['Stage' + str(stage)][train_mode + 'Step'],
                 loss_dict[current_experiment]['Stage' + str(stage)][train_mode + loss])
        plt.ylim([0, 4])

        color_dict = color_constants.color_dict()
        color_keys = ['blue', 'springgreen', 'cyan2', 'sapgreen', 'peacock']
        color_list = [color_dict[color_key] for color_key in color_keys]

        plt.rc('font', family='serif')
        for loss in loss_list:
            fig1, axe1 = plt.subplots(figsize=(15, 8))
            for i_stage, stage in enumerate(stage_list):
                plt.plot(loss_dict[current_experiment]['Stage'+str(stage)][train_mode+'Step'],
                         np.convolve(loss_dict[current_experiment]['Stage'+str(stage)][train_mode+loss],
                                     np.ones((average_length,)) / average_length, mode='same'),
                         color=color_list[i_stage],
                         label='Stage'+str(stage),
                         linewidth=3.0)
            xtick_value = [1e6, 2e6, 3e6, 4e6, 5e6]
            xtick_label = ['1e6', '2e6', '3e6', '4e6', '5e6']
            if loss in ['Huber', 'Loss']:
                ytick_value = [0, 1, 2, 3, 4]
                ytick_label = [str(x) for x in ytick_value]
            else:
                ytick_value = [0, 1, 2, 3, 4]
                ytick_label = [str(x) for x in ytick_value]
            plt.plot([0, xtick_value[-1]], [1, 1], '--', color='k', linewidth=1.5)
            plt.yticks(ytick_value, ytick_label, fontsize=24)
            plt.xticks(xtick_value, xtick_label, fontsize=24)
            plt.ylim([0, ytick_value[-1]])
            plt.xlim([0, 5e6])
            legend1 = plt.legend(prop={'size': 24})
            for i_text, text in enumerate(legend1.get_texts()):
                text.set_color(color_list[i_text])
            plt.title(title_dict[current_experiment]+' '+loss, fontsize=24)
            plt.draw()
            save_address = su.address_generator(setting, 'log_folder')+current_experiment+'_'+loss+'.pdf'
            plt.savefig(save_address)
            plt.close()

    train_mode = 'Validation'
    for loss in loss_list:
        color_dict = color_constants.color_dict()
        color_keys = ['springgreen', 'cyan2', 'blue']
        color_list = [color_dict[color_key] for color_key in color_keys]
        setting = su.initialize_setting(current_experiment_list[0])
        fig1, axe1 = plt.subplots(figsize=(15, 8))
        for i_exp, current_experiment in enumerate(current_experiment_list):
            for i_stage, stage in enumerate(stage_list):
                label = None
                if i_stage == 0:
                    label = title_dict[current_experiment]
                plt.plot(loss_dict[current_experiment]['Stage' + str(stage)][train_mode + 'Step'],
                         np.convolve(loss_dict[current_experiment]['Stage' + str(stage)][train_mode + loss],
                                     np.ones((average_length,)) / average_length, mode='same'),
                         color=color_list[i_exp],
                         label=label,
                         linewidth=3.0)
        xtick_value = [1e6, 2e6, 3e6, 4e6, 5e6]
        xtick_label = ['1e6', '2e6', '3e6', '4e6', '5e6']
        if loss == 'BE':
            ytick_value = [0, 0.3]
        else:
            ytick_value = [0, 1, 2, 3, 4]
        ytick_label = [str(x) for x in ytick_value]
        plt.plot([0, xtick_value[-1]], [1, 1], '--', color='k', linewidth=1.5)
        plt.yticks(ytick_value, ytick_label, fontsize=24)
        plt.xticks(xtick_value, xtick_label, fontsize=24)
        plt.ylim([0, ytick_value[-1]])
        plt.xlim([0, 5e6])
        legend1 = plt.legend(prop={'size': 24})
        for i_text, text in enumerate(legend1.get_texts()):
            text.set_color(color_list[i_text])
        plt.title(loss, fontsize=24)
        plt.draw()
        save_address = su.address_generator(setting, 'log_folder') + current_experiment_list[0] +\
            current_experiment_list[1] + '_' + loss
        plt.savefig(save_address + '.pdf')
        plt.savefig(save_address + '.png')
        plt.close()


def get_backup_folder(setting, stage_list):
    backup_root_folder = su.address_generator(setting, 'result_step_folder', stage_list=stage_list)
    backup_number = 1
    backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    while os.path.isdir(backup_folder):
        backup_number = backup_number + 1
        backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'
    if backup_number > 1:
        backup_number = backup_number - 1
    backup_folder = backup_root_folder + 'backup-' + str(backup_number) + '/'

    setting_address = backup_folder + 'setting.txt'
    network_address = backup_folder + 'network.txt'
    with open(setting_address, 'r') as f:
        setting = json.load(f)
    with open(network_address, 'r') as f:
        network_dict = json.load(f)

    return setting, network_dict


if __name__ == '__main__':
    loss_plot()
