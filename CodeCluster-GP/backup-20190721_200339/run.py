from functions.call_m_densenet_01 import forked_synthesizing_net
from functions.call_m_densenet_02 import call_synth_seg_densenet
from functions.call_synthesize_net_perceptual_net import synthesize_net_perceptual_net
import numpy as np
import tensorflow as tf
fold=10#
np.random.seed(1)
tf.set_random_seed(1)
#
# synth_net=forked_synthesizing_net( data=2,
#                          sample_no=2000000,
#                          validation_samples=396,
#                          no_sample_per_each_itr=1000,
#                          train_tag='', validation_tag='', test_tag='',
#                          img_name='',label_name='', torso_tag='',
#                          log_tag='synth-'+str(fold),min_range=-1000,max_range=3000,
#                          Logs='new_Log/',
#                          fold=fold)
# synth_net.run_net()
mixedup=False
l_regu=None#'l1_regularizer'
synth_seg_densenet=synthesize_net_perceptual_net( data=2,
                         sample_no=2000000,
                         validation_samples=396,
                         no_sample_per_each_itr=1000,
                         train_tag='', validation_tag='', test_tag='',
                         img_name='',label_name='', torso_tag='',
                         log_tag='perceptual-'+str(fold),min_range=-1000,max_range=3000,
                         Logs='ASL_LOG/Log_perceptual/regularization/',
                         fold=fold,Server='DL',newdataset=True,mixedup=mixedup,l_regu=l_regu)
synth_seg_densenet.run_net()