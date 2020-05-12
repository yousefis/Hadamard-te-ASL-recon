import tensorflow as tf
class utils:
    def __init__(self):
        a=1
    def gray2rgb(self,image,tensor_index):
        rgb=image[tensor_index,:,:]
        tf.concat([tf.concat([rgb,rgb],axis=2),rgb],axis=2)
        return rgb