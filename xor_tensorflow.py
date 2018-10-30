import tensorflow as tf
import numpy as np

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
Y = np.array([[0, 1, 1, 0]])
LEARNING_RATE = 0.05
STEP = 200000

x = tf.placeholder(tf.float32, shape=X.shape, name='x-input')
y = tf.placeholder(tf.float32, shape=Y.shape, name='y-input')

theta1 = tf.Variable(tf.random_uniform(X.shape[::-1], -1, 1), name='theta1')
theta2 = tf.Variable(tf.random_uniform(Y.shape[::-1], -1, 1), name='theta2')

bias1 = tf.Variable(tf.zeros([Y.shape[1]]), name='bias1')
bias2 = tf.Variable(tf.zeros([1]), name='bias2')

layer1 = tf.sigmoid(tf.matmul(x, theta1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

cost = tf.reduce_mean(tf.square(output - Y.T))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(STEP):
    sess.run(train_step, feed_dict={x: X, y: Y})
    if i % 500 == 0:
        print('Batch:', i)
        print('Inference:')
        print(sess.run(output, feed_dict={x: X, y: Y}))
        print('cose:', sess.run(cost, feed_dict={x: X, y: Y}))
