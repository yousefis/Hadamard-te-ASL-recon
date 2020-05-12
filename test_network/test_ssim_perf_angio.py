# /srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log/synth-forked_synthesizing_net_rotate-1/

from shutil import copyfile
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from functions.test_dir.check_image_range import *
import os

import numpy as np
import tensorflow as tf
from functions.image_reader.read_data import _read_data
from functions.image_reader.image_class import image_class
from functions.loss.loss_fun import _loss_func
del currentdir, parentdir
from functions.network.densenet_angio_perf import _densenet

import SimpleITK as sitk
eps = 1E-5
plot_tag = 'log'
# densnet_unet_config = [1, 3, 3, 3, 1]
# ct_cube_size = 255
# db_size1 = np.int32(ct_cube_size - 2)
# db_size2 = np.int32(db_size1 / 2)
# db_size3 = np.int32(db_size2 / 2)
# crop_size1 = np.int32(((db_size3 - 2) * 2 + 1.0))
# crop_size2 = np.int32((crop_size1 - 2) * 2 + 1)
in_dim = 101
in_size0 = (0)
in_size1 = np.int8(in_dim)
in_size2 = np.int8(in_size1 )  # conv stack
in_size3 = np.int8((in_size2 - 2))  # level_design1
in_size4 = np.int8(in_size3 / 2 - 2 )  # downsampleing1+level_design2
in_size5 = np.int8(in_size4 / 2 -2)  # downsampleing2+level_design3
# in_size6 = int(in_size5 / 2 -2)  # downsampleing2+level_design3
crop_size0 = (0)
crop_size1 = np.int8(2 * in_size5 + 1)
crop_size2 = np.int8(2 * crop_size1  + 1)
final_layer = crop_size2
label_patchs_size = final_layer
patch_window = in_dim  # 89
#
# in_size0 = (0)
# in_size1 = (input_dim)
# in_size2 = (in_size1)  # conv stack
# in_size3 = ((in_size2))  # level_design1
# in_size4 = int(in_size3 / 2)  # downsampleing1+level_design2
# in_size5 = int(in_size4 / 2 - 4)  # downsampleing2+level_design3
# crop_size0 = (0)
# crop_size1 = (2 * in_size5 + 1)
# crop_size2 = (2 * (crop_size1 - 4) + 1)

gt_cube_size = final_layer


# gap = ct_cube_size - gtv_cube_size


def test_all_nets():
    data = 2

    Server = 'DL'
    if Server == 'DL':
        parent_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/'
        data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/'
    else:
        parent_path = '/exports/lkeb-hpc/syousefi/Code/'
        data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'

    img_name = ''
    label_name = ''

    _rd = _read_data(data=data,
                     img_name=img_name,
                     label_name=label_name,
                     dataset_path=data_path)
    '''read path of the images for train, test, and validation'''
    train_data, validation_data, test_data = _rd.read_data_path()
    # parent_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log/synth-forked_synthesizing_net_rotate-1/'
    # parent_path='/exports/lkeb-hpc/syousefi/Code/ASL_LOG/debug_Log/synth-12/'
    parent_path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/ASL_LOG/ssim_perf_angio/experiment-11-shark/'

    chckpnt_dir = parent_path+'unet_checkpoints/'
    result_path=parent_path+'results/'

    if test_vali == 1:
        test_set = validation_data
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
    img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row1')
    img_row2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row2')
    img_row3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row3')
    img_row4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row4')
    img_row5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row5')
    img_row6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row6')
    img_row7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row7')
    img_row8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='img_row8')
    mri_ph = tf.placeholder(tf.float32,shape=[None, None, None, None, 1],name='mri')
    label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label1')
    label2 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label2')
    label3 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label3')
    label4 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label4')
    label5 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label5')
    label6 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label6')
    label7 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label7')
    label8 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label8')
    label9 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label9')
    label10 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label10')
    label11 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label11')
    label12 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label12')
    label13 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label13')
    label14 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='label14')
    is_training = tf.placeholder(tf.bool, name='is_training')
    input_dim = tf.placeholder(tf.int32, name='input_dim')
    # ave_huber = tf.placeholder(tf.float32, name='huber')

    densenet = _densenet()

    y, degree = densenet.densenet(img_row1=img_row1,
                                  img_row2=img_row2,
                                  img_row3=img_row3,
                                  img_row4=img_row4,
                                  img_row5=img_row5,
                                  img_row6=img_row6,
                                  img_row7=img_row7,
                                  img_row8=img_row8,
                                  input_dim=input_dim,
                                  mri=mri_ph,
                                  is_training=is_training)

    loss_instance = _loss_func()
    labels = []
    labels.append(label1)
    labels.append(label2)
    labels.append(label3)
    labels.append(label4)
    labels.append(label5)
    labels.append(label6)
    labels.append(label7)
    labels.append(label8)
    labels.append(label9)
    labels.append(label10)
    labels.append(label11)
    labels.append(label12)
    labels.append(label13)
    labels.append(label14)

    logits = []
    logits.append(y[:, :, :, :, 0, np.newaxis])
    logits.append(y[:, :, :, :, 1, np.newaxis])
    logits.append(y[:, :, :, :, 2, np.newaxis])
    logits.append(y[:, :, :, :, 3, np.newaxis])
    logits.append(y[:, :, :, :, 4, np.newaxis])
    logits.append(y[:, :, :, :, 5, np.newaxis])
    logits.append(y[:, :, :, :, 6, np.newaxis])

    logits.append(y[:, :, :, :, 7, np.newaxis])
    logits.append(y[:, :, :, :, 8, np.newaxis])
    logits.append(y[:, :, :, :, 9, np.newaxis])
    logits.append(y[:, :, :, :, 10, np.newaxis])
    logits.append(y[:, :, :, :, 11, np.newaxis])
    logits.append(y[:, :, :, :, 12, np.newaxis])
    logits.append(y[:, :, :, :, 13, np.newaxis])
    with tf.name_scope('Loss'):
        loss_dic = loss_instance.loss_selector('SSIM_perf_angio',
                                                    labels=labels, logits=logits,
                                                    angio_SSIM_weight=5)
        cost = tf.reduce_mean(loss_dic["loss"], name="cost")
        cost_angio = tf.reduce_mean(loss_dic["angio_SSIM"], name="angio_SSIM")
        cost_perf = tf.reduce_mean(loss_dic["perf_SSIM"], name="perf_SSIM")


    # ========================================================================
    all_loss = tf.placeholder(tf.float32, name='loss')

    # restore the model
    sess = tf.Session()
    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(chckpnt_dir)
    saver.restore(sess, ckpt.model_checkpoint_path)

    copyfile('./test_synthesize_net2.py',result_path+'/test_synthesize_net2.py')

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
        # init = tf.global_variables_initializer()

    loss = 0
    for img_indx in range(len(test_set)):

        [crush, noncrush, perf, angio,mri_decoded, segmentation_decoded,spacing, direction, origin] = _image_class.read_image_for_test(test_set=test_set, img_indx=img_indx,input_size=in_dim,final_layer=final_layer)


        [loss_train1, perf_loss_tr, angio_loss_tr,
          out, ] = sess.run([cost, cost_perf, cost_angio,
                                         y, ],
                                       feed_dict={img_row1: np.expand_dims(np.expand_dims(crush[ 0],axis=-1),axis=0),
                                                  img_row2: np.expand_dims(np.expand_dims(noncrush[ 1],axis=-1),axis=0),
                                                  img_row3: np.expand_dims(np.expand_dims(crush[2],axis=-1),axis=0),
                                                  img_row4: np.expand_dims(np.expand_dims(noncrush[ 3],axis=-1),axis=0),
                                                  img_row5: np.expand_dims(np.expand_dims(crush[ 4],axis=-1),axis=0),
                                                  img_row6: np.expand_dims(np.expand_dims(noncrush[ 5],axis=-1),axis=0),
                                                  img_row7: np.expand_dims(np.expand_dims(crush[6],axis=-1),axis=0),
                                                  img_row8: np.expand_dims(np.expand_dims(noncrush[ 7],axis=-1),axis=0),
                                                  label1: np.expand_dims(np.expand_dims(perf[0],axis=-1),axis=0),
                                                  label2: np.expand_dims(np.expand_dims(perf[ 1],axis=-1),axis=0),
                                                  label3: np.expand_dims(np.expand_dims(perf[ 2],axis=-1),axis=0),
                                                  label4: np.expand_dims(np.expand_dims(perf[ 3],axis=-1),axis=0),
                                                  label5: np.expand_dims(np.expand_dims(perf[ 4],axis=-1),axis=0),
                                                  label6: np.expand_dims(np.expand_dims(perf[ 5],axis=-1),axis=0),
                                                  label7: np.expand_dims(np.expand_dims(perf[ 6],axis=-1),axis=0),
                                                  label8: np.expand_dims(np.expand_dims(angio[ 0],axis=-1),axis=0),
                                                  label9: np.expand_dims(np.expand_dims(angio[ 1],axis=-1),axis=0),
                                                  label10: np.expand_dims(np.expand_dims(angio[ 2],axis=-1),axis=0),
                                                  label11: np.expand_dims(np.expand_dims(angio[ 3],axis=-1),axis=0),
                                                  label12: np.expand_dims(np.expand_dims(angio[ 4],axis=-1),axis=0),
                                                  label13: np.expand_dims(np.expand_dims(angio[ 5],axis=-1),axis=0),
                                                  label14: np.expand_dims(np.expand_dims(angio[ 6],axis=-1),axis=0),
                                                  is_training: False,
                                                  input_dim: patch_window,
                                                  all_loss: -1.
                                                  })

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
        for i in range(7):
            if i==0:
                os.mkdir(parent_path+'results/'+res_dir+'/GT/')
            sitk_angio=sitk.GetImageFromArray(angio[i])
            sitk_angio.SetDirection(direction=direction)
            sitk_angio.SetOrigin(origin=origin)
            sitk_angio.SetSpacing(spacing=spacing)
            sitk.WriteImage(sitk_angio, parent_path + 'results/' + res_dir + '/GT/angio_' + str(i) + '.mha')

            sitk_perf = sitk.GetImageFromArray(perf[i])
            sitk_perf.SetDirection(direction=direction)
            sitk_perf.SetOrigin(origin=origin)
            sitk_perf.SetSpacing(spacing=spacing)
            sitk.WriteImage(sitk_perf, parent_path + 'results/' + res_dir + '/GT/perf_' + str(i) + '.mha')
        a=1


        # plt.imshow(out[0, int(gt_cube_size / 2), :, :, 0])
        # plt.figure()
        loss += loss_train1
        print('Loss_train: ', loss_train1)

    print('Total loss: ', loss / len(test_set))


if __name__ == "__main__":
    Log = 'EsophagusProject/sythesize_code/Log_perceptual/'
    fold =11
    log_tag = 'experiment-' + str(fold) + '/'
    test_vali = 0
    if test_vali == 1:
        out_dir = log_tag + '/result_vali/'
    else:
        out_dir = log_tag + '/results/'
    test_all_nets()