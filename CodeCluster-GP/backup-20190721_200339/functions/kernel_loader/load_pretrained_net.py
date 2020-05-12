import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from functions.read_data import _read_data

from functions.image_reader.image_class import image_class
from functions.loss.perceptual_loss.nontrainable_fullunet3 import _unet

if __name__=='__main__':
    input_dim = 47
    in_size0 = (0)
    in_size1 = (input_dim)
    in_size2 = (in_size1)  # conv stack
    in_size3 = ((in_size2))  # level_design1
    in_size4 = int(in_size3 / 2)  # downsampleing1+level_design2
    in_size5 = int(in_size4 / 2 - 4)  # downsampleing2+level_design3
    crop_size0 = (0)
    crop_size1 = (2 * in_size5 + 1)
    crop_size2 = (2 * (crop_size1 - 4) + 1)
    final_layer = crop_size2

    data = 2

    Server = 'DL'
    if Server == 'DL':
        data_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/Data-01/BrainWeb_permutation2_low/'
    elif Server == 'Shark':
        data_path = '/exports/lkeb-hpc/syousefi/Synth_Data/BrainWeb_permutation2_low/'

    train_tag = 'train/'
    validation_tag = 'validation/'
    test_tag = 'Esophagus/'

    img_name = ''
    label_name = ''
    fold = 0
    batch_no=1
    patch_window=47
    label_patchs_size=47
    ckpoint_path = '/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log_perceptual/perceptual-5/unet_checkpoints/unet_inter_epoch0_point40600.ckpt-40600'

    img_row1 = tf.placeholder(tf.float32, shape=[batch_no,patch_window,patch_window,patch_window, 1])
    label1 = tf.placeholder(tf.float32, shape=[batch_no,label_patchs_size,label_patchs_size,label_patchs_size   , 1])
    # img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    # label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
    input_dim = tf.placeholder(tf.int32, name='unet_input_dim')
    is_training = tf.placeholder(tf.bool, name='unet_is_training')
    unet=_unet(trainable=False,file_name=ckpoint_path)
    y = unet.unet(img_row1=img_row1, input_dim=input_dim, is_training=is_training)
    _rd = _read_data(data=data, train_tag=train_tag, validation_tag=validation_tag, test_tag=test_tag,
                     img_name=img_name, label_name=label_name, dataset_path=data_path)

    train_data, validation_data, test_data = _rd.read_data_path(fold=fold)
    input_cube_size = 47
    gt_cube_size=input_cube_size
    test_vali = 0
    if test_vali == 1:
        test_set = validation_data
    else:
        test_set = test_data
    img_class = image_class(test_set, bunch_of_images_no=1,
                            is_training=1,
                            patch_window=input_cube_size,
                            sample_no_per_bunch=1,
                            label_patch_size=1,
                            validation_total_sample=0)
    sess = tf.Session()
    # tf.initializers.global_variables()

    loss = 0
    for img_indx in range(len(test_set)):

        crush_noncrush_perf_angio = img_class.read_image_for_test(test_set, img_indx)
        for j in range(7):
            for k in range(2, 4):
                img_size = np.shape(crush_noncrush_perf_angio[k][j])[0]
                input = crush_noncrush_perf_angio[k][j][
                        int(img_size / 2 - input_cube_size / 2):int(img_size / 2 + input_cube_size / 2),
                        int(img_size / 2 - input_cube_size / 2):int(img_size / 2 + input_cube_size / 2),
                        int(img_size / 2 - input_cube_size / 2):int(img_size / 2 + input_cube_size / 2)]

                gt = input
                # [ int(input_cube_size / 2 - gt_cube_size / 2):int(
                #     input_cube_size / 2 + gt_cube_size / 2),
                #               int(input_cube_size / 2 - gt_cube_size / 2):int(
                #                   input_cube_size / 2 + gt_cube_size / 2),
                #               int(input_cube_size / 2 - gt_cube_size / 2):int(
                #                   input_cube_size / 2 + gt_cube_size / 2)
                #               ]
                [ out] = sess.run([ y],
                                                          feed_dict={
                                                              img_row1: np.expand_dims(np.expand_dims(input, -1), 0),
                                                              label1: np.expand_dims(np.expand_dims(gt, -1), 0),
                                                              is_training: False,
                                                              input_dim: input_cube_size,
                                                              })

                plt.imshow(out[0, int(out / 2), :, :, 0])
                plt.figure()
                plt.imshow(gt[int(gt_cube_size / 2), :, :])
    print(ckpoint_path)