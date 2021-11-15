from functions.call_m_densenet_01 import forked_synthesizing_net
from functions.call_m_densenet_02 import call_synth_seg_densenet
from functions.call_synthesize_net_mri import call_synthesize_net_mse_angio_perf

import numpy as np
import tensorflow as tf
fold=1
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

##################################
loadModel=0
newdataset=True
Server='DL'
##################################
synth_seg_densenet=call_synthesize_net_mse_angio_perf( data=2,
                         sample_no=2000000,
                         validation_samples=396,
                         no_sample_per_each_itr=1000,
                         train_tag='', validation_tag='', test_tag='',
                         img_name='',label_name='', torso_tag='',
                         log_tag='experiment-'+str(fold),min_range=-1000,max_range=3000,
                         Logs='ASL_LOG/Workshop_log/MSE/',
                         fold=fold,Server=Server,newdataset=newdataset)
synth_seg_densenet.run_net(loadModel=loadModel)