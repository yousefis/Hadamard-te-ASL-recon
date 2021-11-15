
from functions.multi_stage_perf_angio import multi_stage_net_perf_mri

import numpy as np
import tensorflow as tf
fold=1
np.random.seed(1)
tf.set_random_seed(1)

##################################
loadModel=0
newdataset=True
Server='Shark'
##################################
synth_seg_densenet=multi_stage_net_perf_mri( data=2,
                         sample_no=2000000,
                         validation_samples=396,
                         no_sample_per_each_itr=1000,
                         train_tag='', validation_tag='', test_tag='',
                         img_name='',label_name='', torso_tag='',
                         log_tag='experiment-'+str(fold),min_range=-1000,max_range=3000,
                         Logs='ASL_LOG/Workshop_log/multi_stage/',
                         fold=fold,Server=Server,newdataset=newdataset)
synth_seg_densenet.run_net(loadModel=loadModel)