import tensorflow as tf
import numpy as np
class gradients:
    def compute_sum_pow_gradients(self, tensor, var_list):
        grads = tf.gradients(tensor, var_list)
        grads = [tf.reduce_sum(tf.pow(grad, 2)) if grad is not None else tf.reduce_sum(tf.zeros_like(var))
                 for var, grad in zip(var_list, grads)]
        return grads


    def comput_gradients(self, all_loss, perf_loss, angio_loss, var_list):
        grads_p = tf.gradients(perf_loss, var_list)
        grads_p = [grad if grad is not None else tf.zeros_like(var)
                   for var, grad in zip(var_list, grads_p)]
        slopes_p = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads_p]))

        grads_a = tf.gradients(angio_loss, var_list)

        grads_a = [grad if grad is not None else tf.zeros_like(var)
                   for var, grad in zip(var_list, grads_a)]
        slopes_a = tf.sqrt(tf.reduce_sum([tf.reduce_sum(tf.square(g)) for g in grads_a]))

        tf.summary.scalar('gradients/perfusion', slopes_p)
        tf.summary.scalar('gradients/angiography', slopes_a)

        return grads_p, grads_a
    def sum_gradients(self,grads_p):

        slopes_p = np.sqrt(np.sum([np.sum(np.square(g)) for g in grads_p]))
        return slopes_p
