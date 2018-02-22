import tensorflow as tf



#model parameters

par1 = tf.Variable([-1.0], tf.float32)
par2 = tf.Variable([1.0], tf.float32)
par3 = tf.placeholder(tf.float32)

#Inputs and Output

ActualOutput = par1 * par3 + par2

DesiredOutput = tf.placeholder(tf.float32)

Loss_func = tf.square(ActualOutput - DesiredOutput)
loss = tf.reduce_sum(Loss_func)

init = tf.global_variables_initializer()


sess = tf.Session()
sess.run(init)

print(sess.run(loss, {par3: [1,2,3,4], DesiredOutput: [0, -1, -2, -3]}))
