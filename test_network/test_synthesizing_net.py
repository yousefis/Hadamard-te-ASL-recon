from pandas import ExcelWriter
from pandas import ExcelFile
import pandas as pd
# /srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log/synth-forked_synthesizing_net_rotate-1/
import scipy.spatial

from shutil import copyfile
import os,sys,inspect
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions.network.densenet import _densenet
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from functions.image_reader.read_data import _read_data
from functions.image_reader.image_class import image_class
from functions.loss.loss_fun import _loss_func
from functions.network.forked_densenet import _forked_densenet
del currentdir, parentdir
import time


import SimpleITK as sitk
eps = 1E-5
plot_tag = 'perceptual_net'
# densnet_unet_config = [1, 3, 3, 3, 1]
# ct_cube_size = 255
# db_size1 = np.int32(ct_cube_size - 2)
# db_size2 = np.int32(db_size1 / 2)
# db_size3 = np.int32(db_size2 / 2)
# crop_size1 = np.int32(((db_size3 - 2) * 2 + 1.0))
# crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)

patch_window=101
in_size0 = (0)
in_size1 = int(patch_window)
in_size2 = int(in_size1)  # conv stack
in_size3 = int((in_size2-2))  # level_design1
in_size4 = int(in_size3 / 2-2)  # downsampleing1+level_design2
in_size5 = int(in_size4 / 2-2 )  # downsampleing2+level_design3
crop_size0 = (0)
crop_size1 = int(2 * in_size5 + 1)
crop_size2 = int(2 * (crop_size1 ) + 1)
final_layer=crop_size2
label_patchs_size = final_layer



# gap = ct_cube_size - gtv_cube_size


def test_all_nets(fold):
    data = 2
    Server = 'shark'
    if Server == 'DL':
        parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
        data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/'
    else:
        parent_path = '/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/synth-'+str(fold)+'/'
        data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'

    img_name = ''
    label_name = ''

    _rd = _read_data(data=data,
                     img_name=img_name, label_name=label_name,
                     dataset_path=data_path)
    '''read path of the images for train, test, and validation'''
    train_data, validation_data, test_data = _rd.read_data_path()
    # parent_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log/synth-12/'

    chckpnt_dir = parent_path+'unet_checkpoints/'
    result_path=parent_path+'results/'

    if test_vali == 1:
        test_set = validation_data
    elif test_vali==2:
        test_set = train_data
    else:
        test_set = test_data
    # image=tf.placeholder(tf.float32,shape=[batch_no,patch_window,patch_window,patch_window,1])
    # label=tf.placeholder(tf.float32,shape=[batch_no_validation,label_patchs_size,label_patchs_size,label_patchs_size,2])
    # loss_coef=tf.placeholder(tf.float32,shape=[batch_no_validation,1,1,1])

    # img_row1 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row2 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row3 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row4 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row5 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row6 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row7 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    # img_row8 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    #
    # label1 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label2 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label3 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label4 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label5 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label6 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label7 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label8 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label9 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label10 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label11 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label12 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label13 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    # label14 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size, 1])
    img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])

    label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    is_training = tf.placeholder(tf.bool, name='is_training')
    input_dim = tf.placeholder(tf.int32, name='input_dim')
    # ave_huber = tf.placeholder(tf.float32, name='huber')

    forked_densenet = _forked_densenet()

    y, img_row1, img_row2, img_row3, img_row4, \
    img_row5, img_row6, img_row7, img_row8 = \
        forked_densenet.densenet(img_row1=img_row1, img_row2=img_row2, img_row3=img_row3, img_row4=img_row4,
                                 img_row5=img_row5,
                                 img_row6=img_row6, img_row7=img_row7, img_row8=img_row8, input_dim=input_dim,
                                 is_training=is_training)

    loss_instance = _loss_func()
    with tf.name_scope('averaged_mean_squared_error'):  #
        [averaged_huber, perf_loss, angio_loss] = loss_instance.averaged_huber(label1=label1,
                                                                                    label2=label2,
                                                                                    label3=label3,
                                                                                    label4=label4,
                                                                                    label5=label5,
                                                                                    label6=label6,
                                                                                    label7=label7,
                                                                                    label8=label8,
                                                                                    label9=label9,
                                                                                    label10=label10,
                                                                                    label11=label11,
                                                                                    label12=label12,
                                                                                    label13=label13,
                                                                                    label14=label14,
                                                                                    logit1=y[:, :, :, :, 0, np.newaxis],
                                                                                    logit2=y[:, :, :, :, 1, np.newaxis],
                                                                                    logit3=y[:, :, :, :, 2, np.newaxis],
                                                                                    logit4=y[:, :, :, :, 3, np.newaxis],
                                                                                    logit5=y[:, :, :, :, 4, np.newaxis],
                                                                                    logit6=y[:, :, :, :, 5, np.newaxis],
                                                                                    logit7=y[:, :, :, :, 6, np.newaxis],
                                                                                    logit8=y[:, :, :, :, 7, np.newaxis],
                                                                                    logit9=y[:, :, :, :, 8, np.newaxis],
                                                                                    logit10=y[:, :, :, :, 9,
                                                                                            np.newaxis],
                                                                                    logit11=y[:, :, :, :, 10,
                                                                                            np.newaxis],
                                                                                    logit12=y[:, :, :, :, 11,
                                                                                            np.newaxis],
                                                                                    logit13=y[:, :, :, :, 12,
                                                                                            np.newaxis],
                                                                                    logit14=y[:, :, :, :, 13,
                                                                                            np.newaxis]
                                                                                    )
        cost = tf.reduce_mean(averaged_huber, name="cost")

    # ========================================================================
    ave_loss = tf.placeholder(tf.float32, name='loss')
    ave_loss_perf = tf.placeholder(tf.float32, name='loss_perf')
    ave_loss_angio = tf.placeholder(tf.float32, name='loss_angio')

    average_gradient_perf = tf.placeholder(tf.float32, name='grad_ave_perf')
    average_gradient_angio = tf.placeholder(tf.float32, name='grad_ave_angio')

    # restore the model
    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    copyfile('./test_synthesizing_net.py',result_path+'test_synthesizing_net.py')

    _image_class = image_class(train_data
                               , bunch_of_images_no=1,
                               is_training=1,
                               patch_window=patch_window,
                               sample_no_per_bunch=1,
                               label_patch_size=label_patchs_size,
                               validation_total_sample=0)
    learning_rate = 1E-5
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    loss = 0
    mse_angio=[]
    mse_perf=[]
    test_list=[]
    time_list=[]
    for img_indx in range(len(test_set)):
        crush, noncrush, perf, angio,spacing, direction, origin = _image_class.read_image_for_test(test_set=test_set, img_indx=img_indx,input_size=patch_window,final_layer=final_layer)
        test_set[img_indx][0][0].split('/')

        start=time.time()

        [loss_train1, out] =\
        sess.run([cost, y], feed_dict={
                                          img_row1: np.expand_dims(np.expand_dims(crush[0][:, :, :], axis=0), axis=-1),
                                          img_row2: np.expand_dims(np.expand_dims(noncrush[1][:, :, :], axis=0),
                                                                   axis=-1),
                                          img_row3: np.expand_dims(np.expand_dims(crush[2][:, :, :], axis=0), axis=-1),
                                          img_row4: np.expand_dims(np.expand_dims(noncrush[3][:, :, :], axis=0),
                                                                   axis=-1),
                                          img_row5: np.expand_dims(np.expand_dims(crush[4][:, :, :], axis=0), axis=-1),
                                          img_row6: np.expand_dims(np.expand_dims(noncrush[5][:, :, :], axis=0),
                                                                   axis=-1),
                                          img_row7: np.expand_dims(np.expand_dims(crush[6][:, :, :], axis=0), axis=-1),
                                          img_row8: np.expand_dims(np.expand_dims(noncrush[7][:, :, :], axis=0),
                                                                   axis=-1),
                                          label1: np.expand_dims(np.expand_dims(perf[0], axis=0), axis=-1),
                                          label2: np.expand_dims(np.expand_dims(perf[1], axis=0), axis=-1),
                                          label3: np.expand_dims(np.expand_dims(perf[2], axis=0), axis=-1),
                                          label4: np.expand_dims(np.expand_dims(perf[3], axis=0), axis=-1),
                                          label5: np.expand_dims(np.expand_dims(perf[4], axis=0), axis=-1),
                                          label6: np.expand_dims(np.expand_dims(perf[5], axis=0), axis=-1),
                                          label7: np.expand_dims(np.expand_dims(perf[6], axis=0), axis=-1),
                                          label8: np.expand_dims(np.expand_dims(angio[0], axis=0), axis=-1),
                                          label9: np.expand_dims(np.expand_dims(angio[1], axis=0), axis=-1),
                                          label10: np.expand_dims(np.expand_dims(angio[2], axis=0), axis=-1),
                                          label11: np.expand_dims(np.expand_dims(angio[3], axis=0), axis=-1),
                                          label12: np.expand_dims(np.expand_dims(angio[4], axis=0), axis=-1),
                                          label13: np.expand_dims(np.expand_dims(angio[5], axis=0), axis=-1),
                                          label14: np.expand_dims(np.expand_dims(angio[6], axis=0), axis=-1),
                                          is_training: False,
                                          input_dim: patch_window,
                                          ave_loss: -1,
                                          ave_loss_perf: -1,
                                          ave_loss_angio: -1,
                                          average_gradient_perf: -1,
                                          average_gradient_angio: -1
                                          })
        end = time.time()
        elapsed_time=end-start
        for i in range(np.shape(out)[-1]):
            image=out[0,:,:,:,i]
            sitk_image=sitk.GetImageFromArray(image)
            res_dir=test_set[img_indx][0][0].split('/')[-2]
            if i==0:
                os.mkdir(parent_path+'results/'+res_dir)
            if i<7:
                nm='perf'
            else:
                nm='angi'
            sitk_image.SetDirection(direction=direction)
            sitk_image.SetOrigin(origin=origin)
            sitk_image.SetSpacing(spacing=spacing)
            sitk.WriteImage(sitk_image,parent_path+'results/'+res_dir+'/'+nm+'_'+str(i%7)+'.mha')
            print(parent_path+'results/'+res_dir+'/'+nm+'_'+str(i%7)+'.mha done!')
        mse_p=0
        mse_a=0
        for i in range(7):
            if i==0:
                os.mkdir(parent_path+'results/'+res_dir+'/GT/')
            sitk_angio=sitk.GetImageFromArray(angio[i])
            sitk_angio.SetDirection(direction=direction)
            sitk_angio.SetOrigin(origin=origin)
            sitk_angio.SetSpacing(spacing=spacing)
            sitk.WriteImage(sitk_angio, parent_path + 'results/' + res_dir + '/GT/angio_' + str(i) + '.mha')

            mse_p=np.mean(np.power(out[0, :, :, :, i + 7] - perf[i], 2))
            mse_a = np.mean(np.power(out[0, :, :, :, i] - angio[i], 2))


            sitk_perf = sitk.GetImageFromArray(perf[i])
            sitk_perf.SetDirection(direction=direction)
            sitk_perf.SetOrigin(origin=origin)
            sitk_perf.SetSpacing(spacing=spacing)
            sitk.WriteImage(sitk_perf, parent_path + 'results/' + res_dir + '/GT/perf_' + str(i) + '.mha')

        # plt.imshow(out[0, int(gt_cube_size / 2), :, :, 0])
        # plt.figure()
        loss += loss_train1
        test_list.append(res_dir)
        mse_angio.append(mse_a)
        mse_perf.append(mse_p)
        time_list.append(elapsed_time)
        print('Loss_train: ', loss_train1)
        df = pd.DataFrame({'test_list':test_list,
                           'mse_angio': mse_angio,
                           'mse_perf':mse_perf,
                           'elapsed_time':time_list
                           })
        writer = ExcelWriter(parent_path + 'results/output.xlsx')
        df.to_excel(writer, 'Sheet1', index=False)
        writer.save()
    print('Total loss: ', loss / len(test_set))


if __name__ == "__main__":

    fold = 7
    log_tag = 'synth-' + str(fold) + '/'
    test_vali = 2
    if test_vali == 1:
        out_dir = log_tag + '/result_vali/'
    elif test_vali==2:
        out_dir = log_tag + '/results_tr/'
    else:
        out_dir = log_tag + '/results/'
    test_all_nets(fold)