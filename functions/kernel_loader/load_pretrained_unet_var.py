# import math as math
import os

import tensorflow as tf

from functions.layers.layers import layers
from functions.loss.perceptual_loss.nontrainable_halfunet3 import _unet
from functions.utils.freez_graph import freeze_graph

# import matplotlib.pyplot as plt
save_pb=True
half_unet_graph_Log = 'EsophagusProject/sythesize_code/Log_perceptual/'
half_unet_graph_log_tag = 'perceptual-' + str(4) + '/'
half_unet_graph_chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/' + half_unet_graph_Log + half_unet_graph_log_tag + '/unet_checkpoints/'
# half_unet_graph_trainable = tf.placeholder(tf.bool, name='trainable')
half_unet_graph_ckpt = tf.train.get_checkpoint_state(half_unet_graph_chckpnt_dir)
path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log_perceptual/perceptual-4/'
unet=_unet(trainable=False)
if save_pb:
    if not os.path.exists(path+'/pbs'):
        os.makedirs(path+'/pbs')
    freeze_graph(half_unet_graph_chckpnt_dir, path+'/pbs/{}.pb'.format('jpg'), 'U_y/U_y_conv3d/bias')

#========================
# show all nodes of a graph
AA=[n.name for n in tf.get_default_graph().as_graph_def().node]
for i in AA:
    print(i)

#========================


layers=layers()
X = tf.placeholder(tf.float32, shape=[None, None, None, None, 1],name='synth_img_row1')
conv1 = layers.conv3d(input,
                                       filters=10,
                                       kernel_size=3,
                                       padding='same',
                                       dilation_rate=1,
                                       is_training=True,
                                       trainable='True',
                                       scope='conv1',
                                       reuse='False')
unet.unet(conv1)
#
#
# half_unet_graph = tf.Graph()
#
# with half_unet_graph.as_default():
#     half_unet_graph_Log = 'EsophagusProject/sythesize_code/Log_perceptual/'
#     half_unet_graph_log_tag = 'perceptual-' + str(4) + '/'
#     half_unet_graph_chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/' + half_unet_graph_Log + half_unet_graph_log_tag + '/unet_checkpoints/'
#     # half_unet_graph_trainable = tf.placeholder(tf.bool, name='trainable')
#     half_unet_graph_ckpt = tf.train.get_checkpoint_state(half_unet_graph_chckpnt_dir)
#     half_unet_graph_saver = tf.train.import_meta_graph(half_unet_graph_ckpt.model_checkpoint_path + '.meta')
#     g_a = tf.import_graph_def(half_unet_graph.as_graph_def(), return_elements=['unet_LD_DS1_conv1_conv3d/kernel:0'], name='')
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     second_saver = tf.train.Saver(var_list=g_a)
#     second_saver.restore(sess, './test_dir/test_save.ckpt')
#     a = sess.graph.get_tensor_by_name('a:0')
#     print(sess.run(a))
#
#
# half_unet_graph = tf.Graph()
#
# with half_unet_graph.as_default():
#     half_unet_graph_Log = 'EsophagusProject/sythesize_code/Log_perceptual/'
#     half_unet_graph_log_tag = 'perceptual-' + str(4) + '/'
#     half_unet_graph_chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/' + half_unet_graph_Log + half_unet_graph_log_tag + '/unet_checkpoints/'
#     # half_unet_graph_trainable = tf.placeholder(tf.bool, name='trainable')
#     half_unet_graph_ckpt = tf.train.get_checkpoint_state(half_unet_graph_chckpnt_dir)
#     half_unet_graph_saver = tf.train.import_meta_graph(half_unet_graph_ckpt.model_checkpoint_path + '.meta')
#     # half_unet_graph_saver = tf.train.Saver()
#     variables = half_unet_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#
#     [half_unet_graph.get_tensor_by_name(var.name) for var in variables]
#
#     restored_var = half_unet_graph.get_tensor_by_name(variables)
# with tf.Session() as sess:
#     half_unet_graph_saver.restore(sess, half_unet_graph_ckpt.model_checkpoint_path)
#     variables = half_unet_graph.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
#     values = [sess.run(v) for v in variables]
# print('done')

