import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
if __name__=='__main__':
    patch_window=5
    labels = tf.placeholder(tf.float32, shape=[patch_window,patch_window])
    logit = tf.placeholder(tf.float32, shape=[patch_window,patch_window])

    labels1=[[0.2,0.6,0.7,0.1,0],
             [0.3,0.1,0.1,0,0],
             [0.3,0.1,0,0,0],
             [0.3,0.1,0,0,0],
             [0.3,0,0,0,0]]

    logit1=[[0.2,0.4,0.5,0.1,.1],
             [0.3,0.6,0.1,0.1,0.1],
             [0.5,0.2,0,0.1,0.1],
             [0.2,0.1,0.1,0.1,0.1],
             [0.4,0.3,0.1,0.1,0.1]]


    plt.imshow(labels1, cmap='gray')
    plt.figure()
    plt.imshow(logit1, cmap='gray')


    reshaped_labels=tf.reshape(labels,[-1])
    reshaped_logit=tf.reshape(logit,[-1])
    bg = tf.cast(tf.logical_not(tf.cast(reshaped_labels, tf.bool)), tf.float32)  # background
    img = tf.cast(tf.cast(reshaped_logit, tf.bool), tf.float32)
    False_bg = (bg * img)


    fg = tf.cast(tf.reshape(tf.cast(tf.cast(labels, tf.bool), tf.int8),[-1] ),tf.float32)  # foreground

    False_bg_penalty = tf.reduce_sum(False_bg/ (tf.reduce_sum(bg)))
    False_bg_penalty1 = (False_bg/ (tf.reduce_sum(bg)))
    loss_foreground=tf.reduce_sum((reshaped_logit-reshaped_labels)*(reshaped_logit-reshaped_labels)*fg)/tf.cast(tf.shape(fg)[0],tf.float32)
    loss_foreground1=((reshaped_logit-reshaped_labels)*(reshaped_logit-reshaped_labels)*fg)/tf.cast(tf.shape(fg)[0],tf.float32)
    A=loss_foreground+False_bg_penalty

    # loss=tf.losses.mean_squared_error(
    #             labels=labels,
    #             predictions=logit )
    # round=tf.cast(tf.logical_not(tf.cast(labels1,tf.bool)),tf.int8)
    sess=tf.Session()
    loss_foreground11,False_bg_penalty11=sess.run([loss_foreground1,False_bg_penalty1],feed_dict={labels:labels1,logit:logit1})
    plt.figure()
    plt.imshow(np.reshape(loss_foreground11,[5,5]), cmap='gray')
    plt.figure()
    plt.imshow(np.reshape(False_bg_penalty11, [5, 5]), cmap='gray')
    # plt.show()
    loss_foreground111,False_bg_penalty1,aa=sess.run([loss_foreground,False_bg_penalty,A],feed_dict={labels:labels1,logit:logit1})
    # mse_r=np.reshape(mse,[5,5])



    print(mse)