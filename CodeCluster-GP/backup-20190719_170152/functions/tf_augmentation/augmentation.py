import math
import tensorflow as tf
class augmentation:
    def __init__(self,seed):
        self.seed_no=seed
        self.max_rotate = 18
    def seed(self):
        self.seed_no+=1
        return self.seed_no
    def noisy_input(self, img_rows, is_training):
        noisy_img_rows = []
        #
        with tf.variable_scope("Noise"):
            rnd = tf.greater_equal(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()), 5)[0]

            mean = tf.random_uniform([1], maxval=5, seed=self.seed())
            stdev = tf.random_uniform([1], maxval=3, seed=self.seed())

            for i in range(len(img_rows)):
                noisy_img_rows.append(tf.cond(tf.logical_and(is_training, rnd),
                                              lambda: img_rows[i] + tf.round(tf.random_normal(tf.shape(img_rows[i]),
                                                                                              mean=mean,
                                                                                              stddev=stdev,
                                                                                              seed=self.seed(),
                                                                                              dtype=tf.float32))
                                              , lambda: img_rows[i]))

        return noisy_img_rows

    # ==================================================

    def flip_lr_input(self, img_rows, is_training):
        flip_lr_img_rows = []
        with tf.variable_scope("LR_Flip"):
            rnd = (tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()), 5))[
                0]  # , seed=int(time.time())))
            for i in range(len(img_rows)):
                flip_lr_img_rows.append(tf.cond(tf.logical_and(is_training, rnd),
                                                lambda: tf.expand_dims(
                                                    tf.image.flip_left_right(tf.squeeze(img_rows[i], 4)),
                                                    axis=4)
                                                , lambda: img_rows[i]))

        return flip_lr_img_rows
    # ==================================================

    def rotate_input(self, img_rows, is_training):
        rotate_img_rows = []
        #
        with tf.variable_scope("Rotate"):
            rnd = tf.greater(tf.random_uniform([1], 0, 10, dtype=tf.int32, seed=self.seed()), 5)[
                0]  # , seed=int(time.time())))
            degree_angle = tf.random_uniform([1], minval=-self.max_rotate, maxval=self.max_rotate, seed=self.seed())[0]
            radian = degree_angle * math.pi / 180
            # if rnd:
            for i in range(len(img_rows)):
                rotate_img_rows.append(tf.cond(tf.logical_and(is_training, rnd),
                                               lambda: tf.expand_dims(
                                                   tf.contrib.image.rotate(tf.squeeze(img_rows[i], 4), radian),
                                                   axis=4)
                                               , lambda: img_rows[i]))

        return rotate_img_rows, degree_angle
    # ==================================================

