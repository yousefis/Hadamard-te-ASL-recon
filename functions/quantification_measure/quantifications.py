import numpy as np
import tensorflow as tf
class quantifications:
    def __init__(self):
        a=1
    def quantifiying_gm(self,outputs,mask):
        sum_gm=[]
        for i in range(np.shape(outputs)[0]):
            mean_gm_tmp=0
            gm_ind = np.where(mask[i, :, :, :])
            for j in range(np.shape(outputs)[-1]):
                mean_gm_tmp+=np.sum(outputs[i,gm_ind[0],gm_ind[1],gm_ind[2],j])
            sum_gm.append(mean_gm_tmp)

        return np.mean(sum_gm)

    def seg_sig(self, segmentation, logits):
        one = tf.constant(1, shape=segmentation.shape, dtype=tf.float32)
        two = tf.constant(2, shape=segmentation.shape, dtype=tf.float32)
        three = tf.constant(3, shape=segmentation.shape, dtype=tf.float32)

        wm_mask = tf.cast(tf.equal(segmentation, three), tf.float32)

        gm_mask = tf.cast(tf.equal(segmentation, two), tf.float32)

        csf_mask = tf.cast(tf.equal(segmentation, one), tf.float32)

        WM = tf.multiply(logits, wm_mask)
        GM = tf.multiply(logits, gm_mask)
        CSF = tf.multiply(logits, csf_mask)

        return tf.reduce_mean(WM), tf.reduce_mean(GM), tf.reduce_mean(CSF)

