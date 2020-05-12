import tensorflow as tf

INT = tf.int32

def graph():
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        a = tf.get_variable("a", shape=[1], dtype=INT, initializer=tf.constant_initializer(10))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.save(sess, './test_dir/test_save.ckpt')
        return g


g = graph()

tf.reset_default_graph()

g_a = tf.import_graph_def(g.as_graph_def(), return_elements=['a:0'], name='')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    second_saver = tf.train.Saver(var_list=g_a)
    second_saver.restore(sess, './test_dir/test_save.ckpt')
    a = sess.graph.get_tensor_by_name('a:0')
    print(sess.run(a))