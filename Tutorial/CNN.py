#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 11:30:10 2018

@author: hanxy
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001  # learning rate

# they have been normalized to range(0,1)
mnist = input_data.read_data_sets('./MNIST', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# plot one example
print(mnist.train.images.shape)  # (55000,28*28)
print(mnist.train.labels.shape)  # (55000,10)

tf_x = tf.placeholder(tf.float32, [None, 28 * 28]) / 255.
image = tf.reshape(tf_x, [-1, 28, 28, 1])  # [batch,height,width,channel]
tf_y = tf.placeholder(tf.int32, [None, 10])  # input y

# cnn
conv1 = tf.layers.conv2d(  # shape(28,28,1)
    inputs=image,
    filters=16,
    kernel_size=5,
    strides=1,
    padding='SAME',
    activation=tf.nn.relu
)  # ->>(28,28,16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2
)  # ->>(14,14,16)

conv2 = tf.layers.conv2d(  # shape(14,14,16)
    inputs=pool1,
    filters=32,
    kernel_size=5,
    strides=1,
    padding='SAME',
    activation=tf.nn.relu
)  # ->>(14,14,32)
pool2 = tf.layers.max_pooling2d(
    conv2,
    pool_size=2,
    strides=2
)  # ->>(7,7,32)

flat = tf.reshape(pool2, [-1, 7 * 7 * 32])  # ->>(7*7*32)
output = tf.layers.dense(flat, 10)  # output layer

loss = tf.losses.softmax_cross_entropy(
    onehot_labels=tf_y,
    logits=output
)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

# return (acc,update_op), and create 2 local variavles
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tf_y, axis=1),
    predictions=tf.argmax(output, axis=1)
)[1]

sess = tf.Session()
# the local variables are for accuracy
init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)
sess.run(init_op)

for step in range(6000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, flat],
                                                  {tf_x: test_x, tf_y: test_y})
        print 'Iteration:', step, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_

# print 10 predictions from test data
test_output = sess.run(output, {tf_x: test_x[:100]})
pred_y = np.argmax(test_output, 1)
print 'prediction number', pred_y
print 'real number', np.argmax(test_y[:100], 1)
