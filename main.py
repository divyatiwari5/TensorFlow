import tensorflow as tf

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
c = node2+node1
sess = tf.Session()

File_writer = tf.summary.FileWriter('/home/divya/PycharmProjects/TensorFlow/graph_info', sess.graph )
# print(sess.run([node1, node2]))
# sess.close()

with tf.Session() as sess:
    output = sess.run(c)
    print(output)