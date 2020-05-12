import tensorflow as tf
import numpy as np
from scipy.ndimage import morphology
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

from functions.layers.downsampler import downsampler
from functions.layers.derivative import derivative_LoG
from functions.loss.ssim_loss import *
class _loss_func:
    def __init__(self):
        self.eps = 1e-6
        self.downsampler=downsampler()
        self.max_angio=2.99771428108
        self.max_perf=17.0151833445

    def mean_squared_error(self,labels,logit):
        loss=tf.losses.mean_squared_error(
            labels=labels,
            predictions=logit )
        return loss
    def huber(self,labels,logit):
        loss = tf.losses.huber_loss(labels,logit)
        return loss


    #========================================================================
    #========================================================================
    #========================================================================
    #MSE
    def MSE_group(self, labels, logits):
        loss=0
        loss+= self.mean_squared_error(labels[0], logits[0])
        loss+= self.mean_squared_error(labels[1], logits[1])
        loss+= self.mean_squared_error(labels[2], logits[2])
        loss+= self.mean_squared_error(labels[3], logits[3])
        loss+= self.mean_squared_error(labels[4], logits[4])
        loss+= self.mean_squared_error(labels[5], logits[5])
        loss+= self.mean_squared_error(labels[6], logits[6])
        return loss

    def MSE_perf_angio(self, labels, logits, angio_mse_weight):
        #perf
        perf_mse = self.MSE_group(labels[0:7],logits[0:7])
        #angio
        angio_mse = self.MSE_group(labels[7:14],logits[7:14])
        loss = perf_mse + angio_mse_weight * angio_mse

        loss_dic = {'loss': loss, 'perf_mse': perf_mse, 'angio_mse': angio_mse}
        return loss_dic

    def MSE_perf(self, labels, logits):
        #perf
        perf_mse = self.MSE_group(labels[0:7],logits[0:7])
        loss = perf_mse
        return loss
    #************************************************************************
    #Huber
    def Huber_group(self, labels, logits):
        loss = 0
        loss += self.huber(labels[0], logits[0])
        loss += self.huber(labels[1], logits[1])
        loss += self.huber(labels[2], logits[2])
        loss += self.huber(labels[3], logits[3])
        loss += self.huber(labels[4], logits[4])
        loss += self.huber(labels[5], logits[5])
        loss += self.huber(labels[6], logits[6])
        loss = loss
        return loss

    def Huber_perf_angio(self, labels, logits, weight):
        # perf
        perf_huber = self.Huber_group(labels[0:7], logits[0:7])
        # angio
        angio_huber = self.Huber_group(labels[7:14], logits[7:14])
        loss = perf_huber + weight * angio_huber
        return loss

    def Huber_perf(self, labels, logits):
        # perf
        perf_huber = self.Huber_group(labels[0:7], logits[0:7])
        loss = perf_huber
        return loss
    #************************************************************************
    # SSIM
    def SSIM_group(self, labels, logits,max_val):
        loss = 0

        loss += SSIM(labels[0], logits[0], max_val=max_val)[0]
        loss += SSIM(labels[1], logits[1], max_val=max_val)[0]
        loss += SSIM(labels[2], logits[2], max_val=max_val)[0]
        loss += SSIM(labels[3], logits[3], max_val=max_val)[0]
        loss += SSIM(labels[4], logits[4], max_val=max_val)[0]
        loss += SSIM(labels[5], logits[5], max_val=max_val)[0]
        loss += SSIM(labels[6], logits[6], max_val=max_val)[0]
        loss = 7-(loss+1)/2
        return loss

    def SSIM_perf_angio(self, labels, logits, weight):
        # perf
        perf_SSIM = self.SSIM_group(labels[0:7], logits[0:7],max_val=self.max_perf)
        # angio
        angio_SSIM = self.SSIM_group(labels[7:14], logits[7:14],max_val=self.max_angio)
        loss = perf_SSIM + weight * angio_SSIM

        loss_dic = {'loss': loss, 'angio_SSIM': angio_SSIM, 'perf_SSIM': perf_SSIM}
        return loss_dic

    def SSIM_perf(self, labels, logits):
        # perf
        perf_SSIM = self.SSIM_group(labels[0:7], logits[0:7],max_val=self.max_perf)
        loss_dic = {'loss': perf_SSIM}
        return loss_dic
    #************************************************************************
    # SSIM&Huber
    def SSIM_Huber_perf_angio(self, labels, logits, weight,ssim_huber_weight):
        # perf
        perf_SSIM = self.SSIM_group(labels[0:7], logits[0:7],max_val=self.max_perf)
        perf_Huber= self.Huber_group(labels[0:7], logits[0:7])
        # angio
        angio_SSIM = self.SSIM_group(labels[7:14], logits[7:14],max_val=self.max_angio)
        angio_Huber = self.Huber_group(labels[7:14], logits[7:14])
        loss_perf = ssim_huber_weight * perf_SSIM + perf_Huber
        loss_angio =  angio_SSIM + angio_Huber
        loss = loss_perf + weight * loss_angio
        return loss

    def SSIM_Huber_perf(self, labels, logits,ssim_huber_weight):
        # perf
        perf_SSIM = self.SSIM_group(labels[0:7], logits[0:7],max_val=self.max_perf)
        perf_Huber = self.Huber_group(labels[0:7], logits[0:7])
        # angio
        loss_perf = ssim_huber_weight * perf_SSIM + perf_Huber
        loss = loss_perf
        return loss
    #************************************************************************
    #VGG
    def LPIPS_group(self, labels, logits):
        loss = self.vgg.LPIPS(label=labels, logit=logits)

        return loss


    def vgg_loss_ssim_perf(self, vgg, labels, logits):
        self.vgg = vgg

        loss_perceptual = self.LPIPS_group(labels=labels[0:7], logits=logits[0:7])
        ssim_dic = self.SSIM_perf(labels=labels[0:7], logits=logits[0:7])
        loss_SSIM = ssim_dic['loss']

        loss = loss_perceptual + loss_SSIM

        loss_dic = {'loss': loss, 'loss_perceptual': loss_perceptual, 'loss_SSIM': loss_SSIM}
        return loss_dic

    # ************************************************************************
    # content VGG
    def content_group(self, labels, logits,vgg_angio_weight=1):
        [loss,perf_loss,angio_loss,losses] = self.vgg.content_feature_subtraction(label=labels, logit=logits,vgg_angio_weight=vgg_angio_weight)
        return loss,perf_loss,angio_loss,losses

    def content_pairwise_vgg_feature(self,labels, logits,vgg_angio_weight):
        [loss,perf_loss,angio_loss,losses] = self.content_group(labels=labels, logits=logits,vgg_angio_weight=vgg_angio_weight)
        return loss,perf_loss,angio_loss,losses
    def content_vgg_pairwise_loss_perf(self, vgg, labels, logits,vgg_angio_weight):
        self.vgg = vgg
        loss_perceptual,perf_loss,angio_loss,losses = self.content_pairwise_vgg_feature(labels=labels, logits=logits,vgg_angio_weight=vgg_angio_weight)
        loss_dic = {'loss': loss_perceptual,'perf_loss': perf_loss, 'angio_loss': angio_loss,'losses':losses}
        return loss_dic
    def vgg_loss_perf(self,vgg,labels, logits):
        self.vgg = vgg
        loss_perceptual = self.LPIPS_group(labels[0:7], logits[0:7])
        loss = loss_perceptual
        loss_dic={'loss':loss}
        return loss_dic
    #************************************************************************
    # Huber
    def Huber_g(self, labels, logits):
        loss = []
        loss.append(self.huber(labels[0], logits[0]))
        loss.append(self.huber(labels[1], logits[1]))
        loss.append(self.huber(labels[2], logits[2]))
        loss.append(self.huber(labels[3], logits[3]))
        loss.append(self.huber(labels[4], logits[4]))
        loss.append(self.huber(labels[5], logits[5]))
        loss.append(self.huber(labels[6], logits[6]))
        return loss
    def content_vgg_pairwise_loss_huber(self, vgg, labels, logits,h_labels,h_logits,vgg_angio_weight=1.0,huber_weight=1.0,vgg_weight=1.0):
        if huber_weight:
            # perf
            huber_perf = self.Huber_g(h_labels[0:7], h_logits[0:7])
            # angio
            huber_angio = self.Huber_g(h_labels[7:14], h_logits[7:14])

            huber_loss=tf.reduce_sum(huber_perf)+tf.reduce_sum(huber_angio)
        else:
            huber_perf=tf.zeros(7)
            huber_angio=tf.zeros(7)
            huber_loss=tf.zeros(1)[0]

        if vgg_weight:
            self.vgg = vgg
            [vgg_loss, vgg_perf, vgg_angio, vgg_losses] = self.content_pairwise_vgg_feature(labels=labels, logits=logits,vgg_angio_weight=vgg_angio_weight)
        else:
            vgg_loss=tf.zeros(1)[0]
            vgg_perf=tf.zeros(1)[0]
            vgg_angio=tf.zeros(1)[0]
            vgg_losses=tf.zeros(14)
        #vgg_angio_weight already applied in vgg_feature_maker
        loss_dic = {'loss': vgg_weight*vgg_loss+huber_weight*huber_loss,
                    'vgg_loss':vgg_weight*vgg_loss, #1
                    'vgg_perf': vgg_weight*vgg_perf,#1
                    'vgg_angio': vgg_weight*vgg_angio,#1
                    'vgg_losses': tf.multiply(vgg_weight,vgg_losses),#14 all losses
                    'huber_loss': huber_weight*huber_loss,#1
                    'huber_perf':tf.multiply(huber_weight,huber_perf),#7
                    'huber_angio':tf.multiply(huber_weight,huber_angio),#7
                    'angio_loss':vgg_weight*vgg_angio+huber_weight*tf.reduce_sum(huber_angio),
                    'perf_loss':vgg_weight*vgg_perf+huber_weight*tf.reduce_sum(huber_perf),
                    }
        return loss_dic
    #************************************************************************
    # VGG ssim
    def ssim_group(self, labels, logits):
        loss,mv = self.vgg.ssim_feature_subtraction(label=labels, logit=logits)
        return loss,mv
    def ssim_pairwise_vgg_feature(self, labels, logits):
        loss,mv = self.ssim_group(labels=labels, logits=logits)
        return loss,mv
    def ssim_vgg_pairwise_loss_perf(self, vgg, labels, logits):
        self.vgg = vgg
        loss_perceptual,mv = self.ssim_pairwise_vgg_feature(labels=labels[0:7], logits=logits[0:7])
        loss_dic = {'loss': loss_perceptual,'loss_perceptual': loss_perceptual, 'loss_SSIM': mv}
        return loss_dic
    #************************************************************************
    def ssim_seg(self, segmentation, logits,labels):
        two = tf.constant(2, shape=segmentation.shape, dtype=tf.float32)
        three = tf.constant(3, shape=segmentation.shape, dtype=tf.float32)

        wm_mask = tf.cast(tf.equal(segmentation, three), tf.float32)

        gm_mask = tf.cast(tf.equal(segmentation, two), tf.float32)


        WM_logit = tf.multiply(logits, wm_mask)
        GM_logit = tf.multiply(logits, gm_mask)

        WM_label = tf.multiply(labels, wm_mask)
        GM_label = tf.multiply(labels, gm_mask)


        WM_ssim = self.SSIM_group(logits=WM_logit,labels=WM_label,max_val=self.max_perf)
        GM_ssim = self.SSIM_group(logits=GM_logit,labels=GM_label,max_val=self.max_perf)

        return WM_ssim,GM_ssim

    #************************************************************************
    def mse_seg(self, segmentation, logits,labels):
        two = tf.constant(2, shape=segmentation.shape, dtype=tf.float32)
        three = tf.constant(3, shape=segmentation.shape, dtype=tf.float32)

        wm_mask = tf.cast(tf.equal(segmentation, three), tf.float32)

        gm_mask = tf.cast(tf.equal(segmentation, two), tf.float32)


        WM_logit = tf.multiply(logits, wm_mask)
        GM_logit = tf.multiply(logits, gm_mask)

        WM_label = tf.multiply(labels, wm_mask)
        GM_label = tf.multiply(labels, gm_mask)


        WM_mse = self.MSE_group(logits=WM_logit,labels=WM_label)
        GM_mse = self.MSE_group(logits=GM_logit,labels=GM_label)

        return WM_mse,GM_mse

    #************************************************************************
    def loss_normalized(self, labels, logits,max_val):
        loss = []

        loss.append(SSIM(labels[0], logits[0], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[1], logits[1], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[2], logits[2], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[3], logits[3], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[4], logits[4], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[5], logits[5], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        loss.append(SSIM(labels[6], logits[6], max_val=max_val,axes=[-4,-3,-2,-1])[0])
        return loss
    def normalized_ssim_angio_perf(self,labels,logits):
        #max value for this error is : 14*batch_size
        perf_loss= self.loss_normalized(labels[0:7],logits[0:7],self.max_perf)
        angio_loss= self.loss_normalized(labels[7:14],logits[7:14],self.max_angio)
        max_error1=tf.reduce_max(perf_loss,axis=[0])
        max_error2=tf.reduce_max(angio_loss,axis=[0])

        # max_error = tf.reduce_max([max_error1,max_error2],axis=[0])

        perfloss=tf.reduce_sum(tf.multiply(tf.reduce_sum(perf_loss,0),1/(max_error1+self.eps)))
        angioloss=tf.reduce_sum(tf.multiply(tf.reduce_sum(angio_loss,0),1/(max_error2+self.eps)))


        loss=perfloss+angioloss
        loss_dic = {'loss': loss, 'angio_SSIM': angioloss, 'perf_SSIM': perfloss,'max_error1':max_error1,'max_error2':max_error2}
        return loss_dic


    #************************************************************************
    def Multistage_ssim_perf_angio_loss(self,labels, logits,stage1,stage2):
        # perf
        perf_SSIM = self.SSIM_group(labels[0:7], logits[0:7], max_val=self.max_perf)
        perf_SSIM_s2 = self.SSIM_group(labels[0:7], stage1[0:7], max_val=self.max_perf)
        perf_SSIM_s4 = self.SSIM_group(labels[0:7], stage2[0:7], max_val=self.max_perf)
        # angio
        angio_SSIM = self.SSIM_group(labels[7:14], logits[7:14], max_val=self.max_angio)
        angio_SSIM_s2 = self.SSIM_group(labels[7:14], stage1[7:14], max_val=self.max_angio)
        angio_SSIM_s4 = self.SSIM_group(labels[7:14], stage2[7:14], max_val=self.max_angio)
        w_s1=.2
        w_s2=.05
        loss_s1 = perf_SSIM_s2 #+ angio_SSIM_s2
        loss_s2 = perf_SSIM_s4 #+ angio_SSIM_s4
        loss = perf_SSIM + angio_SSIM + w_s1 * loss_s1+ w_s2 * loss_s2
        loss_dic = {'loss': loss, 'loss_s1': loss_s1, 'loss_s2': loss_s2, 'angio_loss':angio_SSIM,'perf_loss': perf_SSIM+ w_s1 * loss_s1+ w_s2 * loss_s2}
        return loss_dic


    #************************************************************************

    def loss_selector(self,loss_name, labels,logits,
                      angio_mse_weight=None,angio_huber_weight=None,
                      angio_SSIM_weight=None,ssim_huber_weight=None,
                      vgg=None,
                      h_labels=None, h_logits=None,
                      stage1=None,stage2=None):
        if loss_name=='MSE_perf_angio':
            loss = self.MSE_perf_angio(labels,logits,angio_mse_weight)
        elif loss_name=='MSE_perf':
            loss = self.MSE_perf(labels, logits)

        elif loss_name=='Huber_perf_angio':
            loss = self.Huber_perf_angio(labels,logits,angio_huber_weight)
        elif loss_name=='Huber_perf':
            loss = self.Huber_perf(labels,logits)

        elif loss_name=='SSIM_perf_angio':
            loss = self.SSIM_perf_angio(labels,logits,angio_SSIM_weight)
        elif loss_name=='SSIM_perf':
            loss = self.SSIM_perf(labels,logits)

        elif loss_name=='SSIM_Huber_perf_angio':
            loss = self.SSIM_Huber_perf_angio(labels,logits,angio_SSIM_weight,ssim_huber_weight)
        elif loss_name=='SSIM_Huber_perf':
            loss = self.SSIM_Huber_perf(labels,logits,ssim_huber_weight)

        elif loss_name== 'vgg_loss_ssim_perf':
            loss = self.vgg_loss_ssim_perf(vgg,labels, logits)
        elif loss_name== 'content_vgg_pairwise_loss_perf':
            loss = self.content_vgg_pairwise_loss_perf(vgg,labels, logits,vgg_angio_weight=1.0)
        elif loss_name == 'content_vgg_pairwise_loss_huber':
            loss = self.content_vgg_pairwise_loss_huber(vgg,labels, logits,
                                                             h_labels=h_labels, h_logits=h_logits,
                                                        vgg_angio_weight=0.001,huber_weight=0.,vgg_weight=1.0)
        elif loss_name== 'ssim_vgg_pairwise_loss_perf':
            loss = self.ssim_vgg_pairwise_loss_perf(vgg, labels, logits)
        elif loss_name == 'normalized_ssim_angio_perf':
            loss= self.normalized_ssim_angio_perf(labels,logits)
        elif loss_name=='Multistage_ssim_perf_angio_loss':
            loss = self.Multistage_ssim_perf_angio_loss(labels=labels, logits=logits,stage1=stage1,stage2=stage2)
        else:
            loss=0

        return loss


    #========================================================================
    #========================================================================
    #========================================================================
    def vgg_loss_ssim(self,vgg,
                  p_logit1,
                  p_logit2,
                  p_logit3,
                  p_logit4,
                  p_logit5,
                  p_logit6,
                  p_logit7,
                  p_label1,
                  p_label2,
                  p_label3,
                  p_label4,
                  p_label5,
                  p_label6,
                  p_label7,
                  logit1,
                  logit2,
                  logit3,
                  logit4,
                  logit5,
                  logit6,
                  logit7,
                  label1,
                  label2,
                  label3,
                  label4,
                  label5,
                  label6,
                  label7,

                 ):
        self.vgg = vgg
        loss1 = self.vgg.LPIPS(p_logit1, p_label1)
        loss2 = self.vgg.LPIPS(p_logit2, p_label2)
        loss3 = self.vgg.LPIPS(p_logit3, p_label3)
        loss4 = self.vgg.LPIPS(p_logit4, p_label4)
        loss5 = self.vgg.LPIPS(p_logit5, p_label5)
        loss6 = self.vgg.LPIPS(p_logit6, p_label6)
        loss7 = self.vgg.LPIPS(p_logit7, p_label7)

        perceptual_loss=loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

        ssim1, _ = SSIM(label1, logit1, max_val=30)
        ssim2, _ = SSIM(label2, logit2, max_val=30)
        ssim3, _ = SSIM(label3, logit3, max_val=30)
        ssim4, _ = SSIM(label4, logit4, max_val=30)
        ssim5, _ = SSIM(label5, logit5, max_val=30)
        ssim6, _ = SSIM(label6, logit6, max_val=30)
        ssim7, _ = SSIM(label7, logit7, max_val=30)

        ssim_perf =  ( (ssim1 + ssim2 + ssim3 + ssim4 + ssim5 + ssim6 + ssim7))

        ssim =  ssim_perf

        loss = perceptual_loss+ssim
        return loss,ssim_perf,perceptual_loss
    # def vgg_loss(self,vgg,
    #              logit1,
    #              logit2,
    #              logit3,
    #              logit4,
    #              logit5,
    #              logit6,
    #              logit7,
    #              # logit8,
    #              # logit9,
    #              # logit10,
    #              # logit11,
    #              # logit12,
    #              # logit13,
    #              # logit14,
    #              label1,
    #              label2,
    #              label3,
    #              label4,
    #              label5,
    #              label6,
    #              label7,
    #              # label8,
    #              # label9,
    #              # label10,
    #              # label11,
    #              # label12,
    #              # label13,
    #              # label14
    #              ):
    #     self.vgg=vgg
    #
    #     loss1 = self.vgg.LPIPS(logit1, label1)
    #     loss2 = self.vgg.LPIPS(logit2, label2)
    #     loss3 = self.vgg.LPIPS(logit3, label3)
    #     loss4 = self.vgg.LPIPS(logit4, label4)
    #     loss5 = self.vgg.LPIPS(logit5, label5)
    #     loss6 = self.vgg.LPIPS(logit6, label6)
    #     loss7 = self.vgg.LPIPS(logit7, label7)
    #     # loss8 = self.vgg.LPIPS(logit8, label8)
    #     # loss9 = self.vgg.LPIPS(logit9, label9)
    #     # loss10 = self.vgg.LPIPS(logit10, label10)
    #     # loss11 = self.vgg.LPIPS(logit11, label11)
    #     # loss12 = self.vgg.LPIPS(logit12, label12)
    #     # loss13 = self.vgg.LPIPS(logit13, label13)
    #     # loss14 = self.vgg.LPIPS(logit14, label14)
    #     loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 #+ loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14
    #     return loss
    def averaged_mean_squared_error(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14):
        loss1 =self.mean_squared_error(label1, logit1)
        loss2 =self.mean_squared_error(label2, logit2)
        loss3 =self.mean_squared_error(label3, logit3)
        loss4 =self.mean_squared_error(label4, logit4)
        loss5 =self.mean_squared_error(label5, logit5)
        loss6 =self.mean_squared_error(label6, logit6)
        loss7 =self.mean_squared_error(label7, logit7)
        loss8 =self.mean_squared_error(label8, logit8)
        loss9 =self.mean_squared_error(label9, logit9)
        loss10 =self.mean_squared_error(label10, logit10)
        loss11 =self.mean_squared_error(label11, logit11)
        loss12 =self.mean_squared_error(label12, logit12)
        loss13 =self.mean_squared_error(label13, logit13)
        loss14 =self.mean_squared_error(label14, logit14)
        loss=(loss1+loss2+loss3+loss4+loss5+loss6+loss7+1000*(loss8+loss9+loss10+loss11+loss12+loss13+loss14))
        return  loss, (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7), (
                    loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14)
    def averaged_SSIM_huber(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14):
        ssim1,_ =SSIM(label1, logit1,max_val=30)
        ssim2,_ =SSIM(label2, logit2,max_val=30)
        ssim3,_ =SSIM(label3, logit3,max_val=30)
        ssim4,_ =SSIM(label4, logit4,max_val=30)
        ssim5,_ =SSIM(label5, logit5,max_val=30)
        ssim6,_ =SSIM(label6, logit6,max_val=30)
        ssim7,_ =SSIM(label7, logit7,max_val=30)
        ssim8,_ =SSIM(label8, logit8,max_val=30)
        ssim9,_ =SSIM(label9, logit9,max_val=30)
        ssim10,_ =SSIM(label10, logit10,max_val=30)
        ssim11,_ =SSIM(label11, logit11,max_val=30)
        ssim12,_ =SSIM(label12, logit12,max_val=30)
        ssim13,_ =SSIM(label13, logit13,max_val=30)
        ssim14,_ =SSIM(label14, logit14,max_val=30)

        huber1= self.huber(label1, logit1)
        huber2= self.huber(label2, logit2)
        huber3= self.huber(label3, logit3)
        huber4= self.huber(label4, logit4)
        huber5= self.huber(label5, logit5)
        huber6= self.huber(label6, logit6)
        huber7= self.huber(label7, logit7)
        huber8= self.huber(label8, logit8)
        huber9= self.huber(label9, logit9)
        huber10= self.huber(label10, logit10)
        huber11= self.huber(label11, logit11)
        huber12= self.huber(label12, logit12)
        huber13= self.huber(label13, logit13)
        huber14= self.huber(label14, logit14)

        ssim_weight=2
        angio_huber_weight=1

        ssim_perf = ssim_weight*(7-(ssim1 + ssim2 + ssim3 + ssim4 + ssim5 + ssim6 + ssim7))
        ssim_angio = ssim_weight*(7-(ssim8 + ssim9 + ssim10 + ssim11 + ssim12 + ssim13 + ssim14))

        ssim = ssim_angio + ssim_perf

        huber_angio=angio_huber_weight*(huber1+huber2+huber3+huber4+huber5+huber6+huber7)
        huber_perf=huber8+huber9+huber10+huber11+huber12+huber13+huber14

        huber=huber_angio+huber_perf


        perf_loss=ssim_perf+huber_perf
        angio_loss=ssim_angio+huber_angio

        loss = perf_loss+angio_loss

        return loss,ssim,huber,ssim_angio,ssim_perf,huber_angio,huber_perf,perf_loss,angio_loss
    # def MSE_perf_angio(self,label1,label2,label3,label4,label5,label6,label7,
    #                                 label8, label9, label10, label11, label12, label13, label14,
    #                                 logit1, logit2, logit3, logit4, logit5, logit6, logit7,
    #                                 logit8, logit9, logit10, logit11, logit12, logit13,
    #                                 logit14,
    #                                 perf_loss_fm1,perf_loss_fm2,
    #                                 angio_loss_fm1,angio_loss_fm2,
    #                                         ):
    #     loss1 =self.mean_squared_error(label1, logit1)
    #     loss2 =self.mean_squared_error(label2, logit2)
    #     loss3 =self.mean_squared_error(label3, logit3)
    #     loss4 =self.mean_squared_error(label4, logit4)
    #     loss5 =self.mean_squared_error(label5, logit5)
    #     loss6 =self.mean_squared_error(label6, logit6)
    #     loss7 =self.mean_squared_error(label7, logit7)
    #     loss8 =self.mean_squared_error(label8, logit8)
    #     loss9 =self.mean_squared_error(label9, logit9)
    #     loss10 =self.mean_squared_error(label10, logit10)
    #     loss11 =self.mean_squared_error(label11, logit11)
    #     loss12 =self.mean_squared_error(label12, logit12)
    #     loss13 =self.mean_squared_error(label13, logit13)
    #     loss14 =self.mean_squared_error(label14, logit14)
    #
    #     perf=tf.concat((label1,label2),axis=-1)
    #     perf=tf.concat((perf,label3),axis=-1)
    #     perf=tf.concat((perf,label4),axis=-1)
    #     perf=tf.concat((perf,label5),axis=-1)
    #     perf=tf.concat((perf,label6),axis=-1)
    #     perf=tf.concat((perf,label7),axis=-1)
    #     perf_scale2 = self.downsampler.downsampler(perf, down_scale=2, kernel_name='bspline', normalize_kernel=True, a=-.5,
    #                         default_pixel_value=0)
    #     perf_scale4 = self.downsampler.downsampler(perf, down_scale=4, kernel_name='bspline', normalize_kernel=True,
    #                                                a=-.5,
    #                                                default_pixel_value=0)
    #     angio = tf.concat((label8, label9),axis=-1)
    #     angio = tf.concat((angio, label10),axis=-1)
    #     angio = tf.concat((angio, label11),axis=-1)
    #     angio = tf.concat((angio, label12),axis=-1)
    #     angio = tf.concat((angio, label13),axis=-1)
    #     angio = tf.concat((angio, label14),axis=-1)
    #
    #     angio_scale2 = self.downsampler.downsampler(angio, down_scale=2, kernel_name='bspline', normalize_kernel=True,
    #                                                a=-.5,
    #                                                default_pixel_value=0)
    #     angio_scale4 = self.downsampler.downsampler(angio, down_scale=4, kernel_name='bspline', normalize_kernel=True,
    #                                                a=-.5,
    #                                                default_pixel_value=0)
    #
    #
    #
    #     loss_perf_fm1 =self.mean_squared_error(perf_loss_fm2,perf_scale2)
    #     loss_perf_fm2 =self.mean_squared_error(perf_loss_fm1,perf_scale4)
    #     loss_angio_fm1 =self.mean_squared_error(angio_loss_fm2,angio_scale2)
    #     loss_angio_fm2 =self.mean_squared_error(angio_loss_fm1,angio_scale4)
    #
    #
    #
    #
    #
    #     loss=(loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss_perf_fm1+loss_perf_fm2+1000*(loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss_angio_fm1+loss_angio_fm2))\
    #
    #     return loss

    def MSE_perf_forked_angio(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14,
                                    perf_loss_fm1,perf_loss_fm2,
                                    angio_loss_fm1,angio_loss_fm2,
                                            ):
        loss1 =self.mean_squared_error(label1, logit1)
        loss2 =self.mean_squared_error(label2, logit2)
        loss3 =self.mean_squared_error(label3, logit3)
        loss4 =self.mean_squared_error(label4, logit4)
        loss5 =self.mean_squared_error(label5, logit5)
        loss6 =self.mean_squared_error(label6, logit6)
        loss7 =self.mean_squared_error(label7, logit7)
        loss8 =self.mean_squared_error(label8, logit8)
        loss9 =self.mean_squared_error(label9, logit9)
        loss10 =self.mean_squared_error(label10, logit10)
        loss11 =self.mean_squared_error(label11, logit11)
        loss12 =self.mean_squared_error(label12, logit12)
        loss13 =self.mean_squared_error(label13, logit13)
        loss14 =self.mean_squared_error(label14, logit14)

        perf=tf.concat((label1,label2),axis=-1)
        perf=tf.concat((perf,label3),axis=-1)
        perf=tf.concat((perf,label4),axis=-1)
        perf=tf.concat((perf,label5),axis=-1)
        perf=tf.concat((perf,label6),axis=-1)
        perf=tf.concat((perf,label7),axis=-1)
        perf_scale2 = self.downsampler.downsampler(perf, down_scale=2, kernel_name='bspline', normalize_kernel=True, a=-.5,
                            default_pixel_value=0)
        perf_scale4 = self.downsampler.downsampler(perf, down_scale=4, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)
        angio = tf.concat((label8, label9),axis=-1)
        angio = tf.concat((angio, label10),axis=-1)
        angio = tf.concat((angio, label11),axis=-1)
        angio = tf.concat((angio, label12),axis=-1)
        angio = tf.concat((angio, label13),axis=-1)
        angio = tf.concat((angio, label14),axis=-1)

        angio_scale2 = self.downsampler.downsampler(angio, down_scale=2, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)
        angio_scale4 = self.downsampler.downsampler(angio, down_scale=4, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)

        # loss_perf_fm1 =self.mean_squared_error(perf_loss_fm2,perf_scale2)
        # loss_perf_fm2 =self.mean_squared_error(perf_loss_fm1,perf_scale4)
        loss_angio_fm1 =self.mean_squared_error(angio_loss_fm2,angio_scale2)
        loss_angio_fm2 =self.mean_squared_error(angio_loss_fm1,angio_scale4)

        loss=(loss1+loss2+loss3+loss4+loss5+loss6+loss7+1000*(loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss_angio_fm1+loss_angio_fm2))\

        return loss
    def Huber_perf_forked_angio(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14,
                                    angio_loss_fm1,angio_loss_fm2,
                                            ):
        loss1 =self.huber(label1, logit1)
        loss2 =self.huber(label2, logit2)
        loss3 =self.huber(label3, logit3)
        loss4 =self.huber(label4, logit4)
        loss5 =self.huber(label5, logit5)
        loss6 =self.huber(label6, logit6)
        loss7 =self.huber(label7, logit7)
        loss8 =self.huber(label8, logit8)
        loss9 =self.huber(label9, logit9)
        loss10 =self.huber(label10, logit10)
        loss11 =self.huber(label11, logit11)
        loss12 =self.huber(label12, logit12)
        loss13 =self.huber(label13, logit13)
        loss14 =self.huber(label14, logit14)

        perf=tf.concat((label1,label2),axis=-1)
        perf=tf.concat((perf,label3),axis=-1)
        perf=tf.concat((perf,label4),axis=-1)
        perf=tf.concat((perf,label5),axis=-1)
        perf=tf.concat((perf,label6),axis=-1)
        perf=tf.concat((perf,label7),axis=-1)
        perf_scale2 = self.downsampler.downsampler(perf, down_scale=2, kernel_name='bspline', normalize_kernel=True, a=-.5,
                            default_pixel_value=0)
        perf_scale4 = self.downsampler.downsampler(perf, down_scale=4, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)
        angio = tf.concat((label8, label9),axis=-1)
        angio = tf.concat((angio, label10),axis=-1)
        angio = tf.concat((angio, label11),axis=-1)
        angio = tf.concat((angio, label12),axis=-1)
        angio = tf.concat((angio, label13),axis=-1)
        angio = tf.concat((angio, label14),axis=-1)

        angio_scale2 = self.downsampler.downsampler(angio, down_scale=2, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)
        angio_scale4 = self.downsampler.downsampler(angio, down_scale=4, kernel_name='bspline', normalize_kernel=True,
                                                   a=-.5,
                                                   default_pixel_value=0)

        loss_angio_fm1 =self.huber(angio_loss_fm2,angio_scale2)
        loss_angio_fm2 =self.huber(angio_loss_fm1,angio_scale4)

        loss=(loss1+loss2+loss3+loss4+loss5+loss6+loss7+1000*(loss8+loss9+loss10+loss11+loss12+loss13+loss14+loss_angio_fm1+loss_angio_fm2))

        return loss




    def averaged_huber(self, label1, label2, label3, label4, label5, label6, label7,
                       label8, label9, label10, label11, label12, label13, label14,
                       logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                       logit8, logit9, logit10, logit11, logit12, logit13,
                       logit14):
        loss1 = self.huber(label1, logit1)
        loss2 = self.huber(label2, logit2)
        loss3 = self.huber(label3, logit3)
        loss4 = self.huber(label4, logit4)
        loss5 = self.huber(label5, logit5)
        loss6 = self.huber(label6, logit6)
        loss7 = self.huber(label7, logit7)
        loss8 = self.huber(label8, logit8)
        loss9 = self.huber(label9, logit9)
        loss10 = self.huber(label10, logit10)
        loss11 = self.huber(label11, logit11)
        loss12 = self.huber(label12, logit12)
        loss13 = self.huber(label13, logit13)
        loss14 = self.huber(label14, logit14)
        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + 1000 * (
                loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14))
        return loss, (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7), (
                    loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14)

    def soft_dice(self, logits, labels):
        n_classes = 2
        y_pred = tf.reshape(logits, [-1, n_classes])
        y_true = tf.reshape(labels, [-1, n_classes])

        y_pred = tf.nn.softmax(y_pred)

        intersect = tf.reduce_sum(y_pred * y_true, 0)

        denominator = tf.reduce_sum((y_pred), 0) + tf.reduce_sum(y_true, 0)

        dice_scores = (2.0 * intersect) / (denominator+ self.eps)

        return dice_scores
    def calculate_dice(self,logits,labels):
        dice=0
        for i in range(7):
            lgt=logits[:,:,:,:,i][...,tf.newaxis]
            lgt=tf.concat((lgt,1-lgt),axis=-1)


            lbl=labels[:,:,:,:,i][...,tf.newaxis]
            lbl = tf.concat((lbl, 1 - lbl),axis=-1)

            dice=dice+self.soft_dice(logits=lgt, labels=lbl)[0]
        dice/=7
        return dice

    def synth_seg_huber(self, label1, label2, label3, label4, label5, label6, label7,
                       label8, label9, label10, label11, label12, label13, label14,
                       logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                       logit8, logit9, logit10, logit11, logit12, logit13,
                       logit14,seg_angio_logit,seg_angio_label,seg_perf_logit,seg_perf_label):
        loss1 = self.huber(label1, logit1)
        loss2 = self.huber(label2, logit2)
        loss3 = self.huber(label3, logit3)
        loss4 = self.huber(label4, logit4)
        loss5 = self.huber(label5, logit5)
        loss6 = self.huber(label6, logit6)
        loss7 = self.huber(label7, logit7)
        loss8 = self.huber(label8, logit8)
        loss9 = self.huber(label9, logit9)
        loss10 = self.huber(label10, logit10)
        loss11 = self.huber(label11, logit11)
        loss12 = self.huber(label12, logit12)
        loss13 = self.huber(label13, logit13)
        loss14 = self.huber(label14, logit14)

        ave_perf_dice = self.calculate_dice(logits=seg_perf_logit, labels=seg_perf_label)
        ave_angio_dice=self.calculate_dice(logits=seg_angio_logit,labels=seg_angio_label)



        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 +ave_perf_dice+ 1000 * (
            loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14)+ave_angio_dice)
        return loss, loss1, loss2, loss3, loss4, loss5, loss6, loss7, loss8, loss9, loss10, loss11, loss12, loss13, loss14



    def bg_fg_penalty(self,labels,logit):
        reshaped_labels = tf.reshape(labels, [-1])
        reshaped_logit = tf.reshape(logit, [-1])
        bg = tf.cast(tf.logical_not(tf.cast(reshaped_labels, tf.bool)), tf.float32)  # background
        img = tf.cast(tf.cast(reshaped_logit, tf.bool), tf.float32)
        False_bg = (bg * img)
        False_bg_penalty = tf.reduce_sum(False_bg)/tf.cast(tf.shape(bg)[0],tf.float32)

        fg = tf.cast(tf.reshape(tf.cast(tf.cast(labels, tf.bool), tf.int8), [-1]), tf.float32)  # foreground
        loss_foreground = tf.reduce_sum(
            (reshaped_logit - reshaped_labels) * (reshaped_logit - reshaped_labels) * fg )/tf.cast(tf.shape(fg)[0],tf.float32)

        return False_bg_penalty+loss_foreground
    def MSE_remove_bg(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14):
        loss1 =self.bg_fg_penalty(label1, logit1)
        loss2 =self.bg_fg_penalty(label2, logit2)
        loss3 =self.bg_fg_penalty(label3, logit3)
        loss4 =self.bg_fg_penalty(label4, logit4)
        loss5 =self.bg_fg_penalty(label5, logit5)
        loss6 =self.bg_fg_penalty(label6, logit6)
        loss7 =self.bg_fg_penalty(label7, logit7)
        loss8 =self.bg_fg_penalty(label8, logit8)
        loss9 =self.bg_fg_penalty(label9, logit9)
        loss10 =self.bg_fg_penalty(label10, logit10)
        loss11 =self.bg_fg_penalty(label11, logit11)
        loss12 =self.bg_fg_penalty(label12, logit12)
        loss13 =self.bg_fg_penalty(label13, logit13)
        loss14 =self.bg_fg_penalty(label14, logit14)
        loss=(loss1+loss2+loss3+loss4+loss5+loss6+loss7+10*(loss8+loss9+loss10+loss11+loss12+loss13+loss14))
        return loss,loss1,loss2,loss3,loss4,loss5,loss6,loss7,loss8,loss9,loss10,loss11,loss12,loss13,loss14


    def averaged_derivative_huber(self,label1,label2,label3,label4,label5,label6,label7,
                                    label8, label9, label10, label11, label12, label13, label14,
                                    logit1, logit2, logit3, logit4, logit5, logit6, logit7,
                                    logit8, logit9, logit10, logit11, logit12, logit13,
                                    logit14):
        loss1 = self.huber(label1, logit1)
        loss2 = self.huber(label2, logit2)
        loss3 = self.huber(label3, logit3)
        loss4 = self.huber(label4, logit4)
        loss5 = self.huber(label5, logit5)
        loss6 = self.huber(label6, logit6)
        loss7 = self.huber(label7, logit7)
        loss8 = self.huber(label8, logit8)
        loss9 = self.huber(label9, logit9)
        loss10 = self.huber(label10, logit10)
        loss11 = self.huber(label11, logit11)
        loss12 = self.huber(label12, logit12)
        loss13 = self.huber(label13, logit13)
        loss14 = self.huber(label14, logit14)

        derivative_loss1 = self.huber(derivative_LoG(label1), derivative_LoG(logit1))
        derivative_loss2 = self.huber(derivative_LoG(label2), derivative_LoG(logit2))
        derivative_loss3 = self.huber(derivative_LoG(label3), derivative_LoG(logit3))
        derivative_loss4 = self.huber(derivative_LoG(label4), derivative_LoG(logit4))
        derivative_loss5 = self.huber(derivative_LoG(label5), derivative_LoG(logit5))
        derivative_loss6 = self.huber(derivative_LoG(label6), derivative_LoG(logit6))
        derivative_loss7 = self.huber(derivative_LoG(label7), derivative_LoG(logit7))
        derivative_loss8 = self.huber(derivative_LoG(label8), derivative_LoG(logit8))
        derivative_loss9 = self.huber(derivative_LoG(label9), derivative_LoG(logit9))
        derivative_loss10 = self.huber(derivative_LoG(label10), derivative_LoG(logit10))
        derivative_loss11 = self.huber(derivative_LoG(label11), derivative_LoG(logit11))
        derivative_loss12 = self.huber(derivative_LoG(label12), derivative_LoG(logit12))
        derivative_loss13 = self.huber(derivative_LoG(label13), derivative_LoG(logit13))
        derivative_loss14 = self.huber(derivative_LoG(label14), derivative_LoG(logit14))

        derivative_loss = derivative_loss1+derivative_loss2+derivative_loss3+derivative_loss4+derivative_loss5+derivative_loss6+\
        derivative_loss7+derivative_loss8+derivative_loss9+derivative_loss10+derivative_loss11+derivative_loss12+derivative_loss13+derivative_loss14

        loss = (loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 +  1000*(loss8 + loss9 + loss10 + loss11 + loss12 + loss13 + loss14))
        all_loss=loss+derivative_loss
        return all_loss, derivative_loss,derivative_loss


