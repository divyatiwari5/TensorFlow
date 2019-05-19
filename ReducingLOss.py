import tensorflow as tf

#model parameters
par1 = tf.Variable([1.0], tf.float32)
par2 = tf.Variable([-1.0], tf.float32)
par3 = tf.placeholder(tf.float32)

#Inputs and Output
ActualOutput = par1 * par3 + par2

DesiredOutput = tf.placeholder(tf.float32)

#Loss
Loss_func = tf.square(ActualOutput - DesiredOutput)
loss = tf.reduce_sum(Loss_func)

#Optimize
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#Add an operation to initialize a variable
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range (1000):
    sess.run(train, {par3: [1,2,3,4], DesiredOutput: [0,-1,-2,-3]})

print(sess.run([par1, par2]))