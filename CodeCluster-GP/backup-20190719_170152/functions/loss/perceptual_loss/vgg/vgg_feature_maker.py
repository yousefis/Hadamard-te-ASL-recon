import tensorflow as tf

from functions.loss.perceptual_loss.vgg.vgg_loader import vgg_loader
from functions.loss.perceptual_loss.vgg.vgg_utils import *
from os.path import dirname, abspath

from functions.loss.ssim_loss import *
class vgg_feature_maker:
    def __init__(self,test=0):
        parent_path = dirname(dirname(abspath(__file__)))

        self.data_dict = loadWeightsData(parent_path + '/vgg/pretrained_vgg/vgg16.npy')
        # self.gram = []
        self.vgg_loader=vgg_loader(data_dict=self.data_dict)
        self.test=test

    def feed_img(self, input,feature_type='huber'):
        features = self.feed_3dimage_to_2dvgg(input,feature_type=feature_type)
        return features

    def feed_3dimage_to_2dvgg(self, input,feature_type='huber'):
        features=[]
        print('feed_3dimage_to_2dvgg start')
        for i in range(input.get_shape()[1]):
            # r=tf.expand_dims(tf.squeeze(input[:, i, :, :]), 3)
            # g=tf.expand_dims(tf.squeeze(input[:, i, :, :]), 3)
            # b=tf.expand_dims(tf.squeeze(input[:, i, :, :]), 3)
            if self.test==0:
                rgb = tf.tile(tf.expand_dims(tf.squeeze(input[:, :, :, i]), 3),[1,1,1,3])
            else:
                rgb = tf.tile(tf.expand_dims((input[:, :, :, i]), 3), [1, 1, 1, 3])

            # VGG_MEAN = [103.939, 116.779, 123.68]
            # bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)


            self.vgg_loader.vgg_feed(rgb)

            features.append([input,
                             self.vgg_loader.conv1_1,
                             self.vgg_loader.conv1_2,
                             # self.vgg_loader.pool1,
                             # self.vgg_loader.conv2_1,
                             # self.vgg_loader.conv2_2,
                             # self.vgg_loader.pool2,
                             # self.vgg_loader.conv3_1,
                             # self.vgg_loader.conv3_2,
                             # self.vgg_loader.conv3_3,
                             # self.vgg_loader.pool3,
                             # self.vgg_loader.conv4_1,
                             # self.vgg_loader.conv4_2,
                             # self.vgg_loader.conv4_3,
                             # self.vgg_loader.pool4,
                             # self.vgg_loader.conv5_1,
                             # self.vgg_loader.conv5_2,
                             # self.vgg_loader.conv5_3,
                             # self.vgg_loader.pool5
                             ])
        if feature_type=='gram':
            gram = [[self.gram_matrix(features[slice][f]) for slice in range(np.shape(features)[0])] for f in
             range(np.shape(features)[1])]
            all_features=gram
        elif feature_type=='huber':
            # huber = [[self.huber_matrix(features[slice][f]) for slice in range(np.shape(features)[0])] for f in
            #         range(np.shape(features)[1])]
            all_features = features

        print('feed_3dimage_to_2dvgg end')
        return all_features


    def loop_body(self, i, input):
        img = tf.squeeze(input[:, i, :, :, 1])
        rgb = tf.concat([img, tf.concat([img, img], axis=3)], axis=3)

        vgg_s = self.vgg_loader.vgg_feed(rgb=rgb)
        feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
        gram = [self.gram_matrix(l) for l in feature_]
        return gram

    def huber_matrix(self,x):
        assert isinstance(x, tf.Tensor)
        b, h, w, ch = x.get_shape().as_list()
        features = tf.reshape(x, [b, h , ch* w])
        huber= features
        return huber

    # gram matrix per layer
    def gram_matrix(self, x):
        assert isinstance(x, tf.Tensor)
        b, h, w, ch = x.get_shape().as_list()
        features = tf.reshape(x, [b, h * w, ch])
        # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
        gram = tf.matmul(features, features, adjoint_a=True) / tf.constant(ch * w * h, tf.float32)
        return gram

    def compute_loss(self, gram_label):
        batchsize = gram_label.get_shape()[0]
        weight = 1
        # compute style loss
        loss_s = tf.zeros(batchsize, tf.float32)
        for g, g_ in zip(self.gram, gram_label):
            loss_s += weight * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])

    def LPIPS(self, logit, label):
        return tf.reduce_mean(
            [tf.reduce_mean(tf.subtract(logit[i][-1] , label[i][-1]) ** 2) for i in range(len(logit)) ])

    def content_feature_subtraction(self, logit, label,vgg_angio_weight=1.0):
        timepoints_weight=[2,2,2,1,1,1,2,
                           2,2,2,1,1,1,2]
        feature_subtraction=[]
        losses=[]
        for i in range(len(logit)):
            for j in range(len(logit[i])):
                for k in range(len(logit[i][j])):
                    subtracted_f=(tf.subtract(logit[i][j][k], label[i][j][k]) ** 2)
                    feature_subtraction.append(tf.reduce_sum(subtracted_f))
            if i>6:
                angio_weight=vgg_angio_weight
            else:
                angio_weight=1
            angio_weight=angio_weight*timepoints_weight[i]
            losses.append((angio_weight*tf.reduce_sum(feature_subtraction[i*(len(logit[i])*len(logit[i][j])):(i+1)*(len(logit[i])*len(logit[i][j]))])))

        # feature_subtraction= [tf.reduce_mean(tf.subtract(logit[i][-1], label[i][-1]) ** 2) for i in range(len(logit))]
        return tf.reduce_sum(losses),tf.reduce_sum(losses[0:7]),tf.reduce_sum(losses[7:14]),losses

    def ssim_feature_subtraction(self, logit, label):
        feature_subtraction = []
        for i in range(len(logit)):
            max_val=50

            subtracted_f,subtracted_map = GPU_SSIM(logit[i][-1], label[i][-1],max_val=max_val)
            feature_subtraction.append(tf.multiply(tf.divide(subtracted_f, tf.cast(
                subtracted_map.shape[0] * subtracted_map.shape[1] * subtracted_map.shape[2] * subtracted_map.shape[3],
                tf.float32)),10E7))
            max_val1=tf.reduce_max(logit[i][-1])
            max_val2 = tf.reduce_max(label[i][-1])
            max_val_=tf.reduce_max([max_val1,max_val2])
        return tf.reduce_sum(feature_subtraction),max_val_

