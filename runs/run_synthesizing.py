from functions.call_synthesizing_net import synthesize_net
from functions.call_synthesize_net_perceptual_net import synthesize_net_perceptual_net
from functions.call_forked_synthesizing_net import forked_synthesizing_net
from functions.call_forked_synthesizing_net2 import forked_synthesizing_net2
# from functions.call_forked_synthesizing_net_derivativeloss import forked_synthesizing_net
from functions.call_forked_synthesizing_net22 import forked_synthesizing_net22
import numpy as np
import tensorflow as tf
fold=12
np.random.seed(1)
tf.set_random_seed(1)

dc12=forked_synthesizing_net( data=2,
                             sample_no=2000000,
                             validation_samples=396,
                             no_sample_per_each_itr=1000,
                             train_tag='', validation_tag='', test_tag='',
                             img_name='',label_name='', torso_tag='',
                             log_tag='synth-'+str(fold),min_range=-1000,max_range=3000,
                             Logs='ASL_LOG/debug_Log/',
                             fold=fold)

dc12.run_net()


