import numpy as np
import tensorflow as tf

from functions.loss.perceptual_loss.network.half_unet import _half_unet


#https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11

class perceptual_loss:
    def __init__(self):
        self.input_cube_size = 53
        self.in_size0 = (0)
        self.in_size1 = (self.input_cube_size)
        self.in_size2 = (self.in_size1)  # conv stack
        self.in_size3 = ((self.in_size2))  # level_design1
        self.in_size4 = int(self.in_size3 / 2)  # downsampleing1+level_design2
        self.in_size5 = int(self.in_size4 / 2 - 4)  # downsampleing2+level_design3
        self.crop_size0 = (0)
        self.crop_size1 = (2 * self.in_size5 + 1)
        self.crop_size2 = (2 * (self.crop_size1 - 4) + 1)
        self.final_layer = self.crop_size2
        self.fold=2
        self.gt_cube_size = self.final_layer
        self.log_tag = 'perceptual-' + str(self.fold) + '/'
        self.Log = 'EsophagusProject/sythesize_code/Log_perceptual/'
        self.chckpnt_dir = '/srv/2-lkeb-17-dl01/syousefi/TestCode/' + self.Log + self.log_tag + '/unet_checkpoints/'
        self.test_vali=0
        #https://blog.metaflow.fr/tensorflow-saving-restoring-and-mixing-multiple-models-c4c94d5d7125

        self.ckpt = tf.train.get_checkpoint_state(self.chckpnt_dir)

        #
        # saver = tf.train.import_meta_graph( self.chckpnt_dir+ '/unet_inter_epoch0_point9700.ckpt-9700.meta')
        # saver =tf.train.import_meta_graph(self.ckpt.model_checkpoint_path + '.meta')
        # graph = tf.get_default_graph()
        # input = graph.get_tensor_by_name('cond_1/Merge:0')
        # level_us1 = graph.get_tensor_by_name('US1/cond_1/Merge:0')
        # output_conv_sg = tf.stop_gradient(level_us1)


        self.trainable = tf.placeholder(tf.bool, name='trainable')
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.ave_huber = tf.placeholder(tf.float32, name='huber')
        # self.img_row1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        self.img_row1 = tf.placeholder(tf.float32, shape=[1, 53, 53, 53, 1])
        self.label1 = tf.placeholder(tf.float32, shape=[None, None, None, None, 1])
        self.input_dim = tf.placeholder(tf.int32, name='input_dim')
        self.half_unet = _half_unet(trainable=self.trainable)
        #
        # self.level_us1 = self.half_unet.half_unet(img_row1=self.img_row1, input_dim=self.input_dim, is_training=self.is_training)
        #
        # # tf.get_default_graph().get_tensor_by_name('US1/cond_1/Merge:0')
        # output_conv_sg = tf.stop_gradient(self.level_us1)  # It's an identity function
        #
        # self.sess = tf.Session()
        # self.saver = tf.train.Saver()
        # # with tf.name_scope('Unet'):
        # self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
        # # next_to_last_tensor = self.sess.graph.get_tensor_by_name('US1')
        #
        #
        # self.graph = tf.get_default_graph()
        # # self.US1 = self.graph.get_tensor_by_name("US1:0")  # Tensor to import
        # # for i in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='US1'):
        # #     print(i)   # i.name if you want just a name
        # self.loss_func=_loss_func()

        # ================================================
        # removes trainable variables:
        trainable_collection = tf.get_collection_ref(tf.GraphKeys.TRAINABLE_VARIABLES)
        for i in range(len(trainable_collection)):
            trainable_collection.pop(0)
        # ================================================

    def unet_features(self, preceptual_input, perceptual_gt):


        # averaged_mse = self.loss_instance.mean_squared_error(labels=self.label1,
        #                                                 logit=y[:, :, :, :, 0, np.newaxis])
        # cost = tf.reduce_mean(averaged_mse, name="cost")
        # restore the model

        if tf.is_numeric_tensor(preceptual_input):
            [unet_end]=self.sess.run([self.level_us1],
                          feed_dict={self.img_row1: np.zeros(
                              [1, self.input_cube_size, self.input_cube_size, self.input_cube_size, 1]),
                                     self.label1: np.zeros(
                                         [1, self.gt_cube_size, self.gt_cube_size, self.gt_cube_size, 1]),
                                     self.is_training: False,
                                     self.input_dim: self.input_cube_size,
                                     self.ave_huber: -1,
                                     self.trainable: False
                                     })
        else:
            [unet_end] = self.sess.run([self.level_us1],
                                      feed_dict={self.img_row1: preceptual_input,
                                                 self.label1: perceptual_gt,
                                                 self.is_training: False,
                                                 self.input_dim: self.input_cube_size,
                                                 self.ave_huber: -1,
                                                 self.trainable: False
                                                 })
        return unet_end#, unet_end
    def perceptual_loss(self,main_input, main_gt):
        perceptual_gt=main_input[int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
                            self.input_cube_size / 2 + self.gt_cube_size / 2),
                        int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
                            self.input_cube_size / 2 + self.gt_cube_size / 2),
                        int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
                            self.input_cube_size / 2 + self.gt_cube_size / 2)
                        ]
        US_input=self.unet_features(preceptual_input=main_input,perceptual_gt=perceptual_gt)

        perceptual_gt = main_gt[int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
            self.input_cube_size / 2 + self.gt_cube_size / 2),
                        int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
                            self.input_cube_size / 2 + self.gt_cube_size / 2),
                        int(self.input_cube_size / 2 - self.gt_cube_size / 2):int(
                            self.input_cube_size / 2 + self.gt_cube_size / 2)
                        ]
        US_gt = self.unet_features(preceptual_input=main_gt, perceptual_gt=perceptual_gt)
        return US_input,US_gt
    def main_net_perceptual_loss_perf(self,perf_net1,perf_net2,perf_net3,perf_net4,perf_net5,perf_net6,perf_net7,
                                 perf_gt1, perf_gt2, perf_gt3, perf_gt4, perf_gt5, perf_gt6, perf_gt7):
        US_input1, US_gt1=self.perceptual_loss( perf_net1, perf_gt1)
        [US_input2, US_gt2]=self.perceptual_loss( perf_net2, perf_gt2)
        [US_input3, US_gt3]=self.perceptual_loss( perf_net3, perf_gt3)
        [US_input4, US_gt4]=self.perceptual_loss( perf_net4, perf_gt4)
        [US_input5, US_gt5]=self.perceptual_loss( perf_net5, perf_gt5)
        [US_input6, US_gt6]=self.perceptual_loss( perf_net6, perf_gt6)
        [US_input7, US_gt7]=self.perceptual_loss( perf_net7, perf_gt7)

        loss1=self.loss_func.huber(US_input1, US_gt1)
        loss2=self.loss_func.huber(US_input2, US_gt2)
        loss3=self.loss_func.huber(US_input3, US_gt3)
        loss4=self.loss_func.huber(US_input4, US_gt4)
        loss5=self.loss_func.huber(US_input5, US_gt5)
        loss6=self.loss_func.huber(US_input6, US_gt6)
        loss7=self.loss_func.huber(US_input7, US_gt7)
        sum_huber=(loss1+loss2+loss3+loss4+loss5+loss6+loss7)
        return sum_huber
    def main_net_perceptual_loss_angio(self,angio_net1,angio_net2,angio_net3,angio_net4,angio_net5,angio_net6,angio_net7,
                                 angio_gt1, angio_gt2, angio_gt3, angio_gt4, angio_gt5, angio_gt6, angio_gt7):


        loss1=self.loss_func.huber(angio_net1, angio_gt1)
        loss2=self.loss_func.huber(angio_net2, angio_gt2)
        loss3=self.loss_func.huber(angio_net3, angio_gt3)
        loss4=self.loss_func.huber(angio_net4, angio_gt4)
        loss5=self.loss_func.huber(angio_net5, angio_gt5)
        loss6=self.loss_func.huber(angio_net6, angio_gt6)
        loss7=self.loss_func.huber(angio_net7, angio_gt7)
        sum_huber=(loss1+loss2+loss3+loss4+loss5+loss6+loss7)
        return sum_huber

    def main_net_perceptual_loss(self,perf_net1,perf_net2,perf_net3,perf_net4,perf_net5,perf_net6,perf_net7,
                                 angio_net1, angio_net2, angio_net3, angio_net4, angio_net5, angio_net6, angio_net7,
                                 perf_gt1, perf_gt2, perf_gt3, perf_gt4, perf_gt5, perf_gt6, perf_gt7,
                                 angio_gt1, angio_gt2, angio_gt3, angio_gt4, angio_gt5, angio_gt6, angio_gt7):
        sum_huber_perf=self.main_net_perceptual_loss_perf( perf_net1, perf_net2, perf_net3, perf_net4, perf_net5, perf_net6, perf_net7,
                                      perf_gt1, perf_gt2, perf_gt3, perf_gt4, perf_gt5, perf_gt6, perf_gt7)
        sum_huber_angio =self.main_net_perceptual_loss_angio( angio_net1, angio_net2, angio_net3, angio_net4, angio_net5, angio_net6,
                                       angio_net7,angio_gt1, angio_gt2, angio_gt3, angio_gt4, angio_gt5, angio_gt6, angio_gt7)
        average_huber=(sum_huber_perf+sum_huber_angio)
        return average_huber