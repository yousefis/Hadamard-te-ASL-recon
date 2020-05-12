import json
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
if __name__=='__main__':
    log_name_train_type = {'Training': 'train/',
                                   'Validation': 'validation/'}
    train_mode_list = ['Training']#,Validation
    tag_list = ['gradients/avegare_perfusion', 'gradients/perfusion',
                 'gradients/avegare_angiography','gradients/angiography',
                'Loss/ave_loss', 'Loss/ave_loss_angio', 'Loss/ave_loss_perf']

    for train_mode in train_mode_list:
        log_folder='/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/synth-5/'
        log_test_folder = log_folder + log_name_train_type[train_mode]

        loss_dict = dict()
        file_list = [f for f in os.listdir(log_test_folder) if os.path.isfile(os.path.join(log_test_folder, f))]

        for file in file_list:
            step = []
            g_ave_perf = []
            g_perf = []
            g_ave_angio = []
            g_angio = []
            loss = []
            loss_a_ave = []
            loss_p_ave = []
            i=0
            options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
            for e in tf.train.summary_iterator(log_test_folder + file):
                print(e)
            for e  in tf.train.summary_iterator(log_test_folder + file):
                print(i)
                i+=1
                if i==1111:
                    continue
                for v in e.summary.value:
                    try:
                        step.append(e.step)
                        if v.tag == tag_list[0]:
                            g_ave_perf.append(v.simple_value)
                        elif v.tag == tag_list[1]:
                            g_perf.append(v.simple_value)
                        elif v.tag == tag_list[2]:
                            g_ave_angio.append(v.simple_value)
                        elif v.tag == tag_list[3]:
                            g_angio.append(v.simple_value)
                        elif v.tag == tag_list[4]:
                            loss.append(v.simple_value)
                        elif v.tag == tag_list[5]:
                            loss_a_ave.append(v.simple_value)
                        elif v.tag == tag_list[6]:
                            loss_p_ave.append(v.simple_value)
                    except:
                        a=1

            print('file')

