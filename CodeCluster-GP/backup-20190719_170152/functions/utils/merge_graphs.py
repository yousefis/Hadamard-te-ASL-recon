import tensorflow as tf

path='/srv/2-lkeb-17-dl01/syousefi/TestCode/EsophagusProject/sythesize_code/Log/merge_gpah/'
# 1. Create and save two graphs

# c = a*b
g1 = tf.Graph()
with g1.as_default():
    a = tf.placeholder(tf.float32, name='a')
    b = tf.Variable(initial_value=tf.truncated_normal((1,)), name='b')
    c = tf.multiply(a, b, name='c')
    s1 = tf.train.Saver()

with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    b_init = sess.run(b)
    s1.save(sess, path+'g1')


# f = d*e
g2 = tf.Graph()
with g2.as_default():
    d = tf.placeholder(tf.float32, name='d')
    e = tf.Variable(initial_value=tf.truncated_normal((1,)), name='e')
    f = tf.multiply(d, e, name='f')
    s2 = tf.train.Saver()

with tf.Session(graph=g2) as sess:
    sess.run(tf.global_variables_initializer())
    e_init = sess.run(e)
    s2.save(sess, path+'g2')

# 2.A Stack loaded models horizontally: g = a*b + d*e

g3 = tf.Graph()
with g3.as_default():
    tf.train.import_meta_graph(path+'g1.meta', import_scope='g1')
    a_, b_, c_ = [g3.get_tensor_by_name('g1/%s:0' % name) for name in ('a', 'b', 'c')]

    tf.train.import_meta_graph(path+'g2.meta', import_scope='g2')
    d_, e_, f_ = [g3.get_tensor_by_name('g2/%s:0' % name) for name in ('d', 'e', 'f')]

    g = c_ + f_

# create separate loaders - we need to load variables from different files
with g3.as_default():
    s31 = tf.train.Saver(var_list={'b': b_})
    s32 = tf.train.Saver(var_list={'e': e_})


feed_dict = {a_: 1.0, d_: -1.0}
print('a=%s and d=%s' % (feed_dict[a_], feed_dict[d_]))

with tf.Session(graph=g3) as sess:
    s31.restore(sess, path+'g1')
    s32.restore(sess, path+'g2')

    # check if values were actually restored, not re-initialized
    b_value, e_value, g_value = sess.run([b_, e_, g], feed_dict=feed_dict)
    assert b_init == b_value, 'saved %s and restored %s' % (b_init, b_value)
    assert e_init == e_value, 'saved %s and restored %s' % (e_init, e_value)
    print('restored %s and %s ' % (b_value, e_value))

    # check if model works correctly
    assert g_value == feed_dict[a_] * b_init + feed_dict[d_] * e_init
    print('a*b + d*e = %s' % g_value)


# 2.B Stack loaded models vertically: g = e*(a*b)
g4 = tf.Graph()

with g4.as_default():
    tf.train.import_meta_graph(path+'g1.meta', import_scope='g1')
    a_, b_, c_ = [g4.get_tensor_by_name('g1/%s:0' % name) for name in ('a', 'b', 'c')]

    tf.train.import_meta_graph(path+'g2.meta', import_scope='g2', input_map={'d:0': c_})
    e_, g = [g4.get_tensor_by_name('g2/%s:0' % name) for name in ('e', 'f')]

# create separate loaders again
with g4.as_default():
    s41 = tf.train.Saver(var_list={'b': b_})
    s42 = tf.train.Saver(var_list={'e': e_})

feed_dict = {a_: 1.0}
print('a=%s' % feed_dict[a_])


with tf.Session(graph=g4) as sess:
    s41.restore(sess, path+'g1')
    s42.restore(sess, path+'g2')

    # check restored values
    b_value, e_value, g_value = sess.run([b_, e_, g], feed_dict=feed_dict)
    assert b_init == b_value, 'saved %s and restored %s' % (b_init, b_value)
    assert e_init == e_value, 'saved %s and restored %s' % (e_init, e_value)
    print('restored %s and %s ' % (b_value, e_value))

    # check if model works correctly
    assert g_value == feed_dict[a_] * b_init * e_init
    print('e*(a*b) = %s' % g_value)