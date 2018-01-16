#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 17:34:28 2018

@author: hanxy
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
TIME_STEP = 28      # rnn time step / image height
INPUT_SIZE = 28     # rnn input size / image width
LR = 0.01           # learning rate

# data
mnist = input_data.read_data_sets('MNIST', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE])   # shape(batch, 28*28)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])               # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
        rnn_cell,                       # cell you have chosen
        image,                          # input
        initial_state=None,             # the initial hidden state
        dtype=tf.float32,               # must given if set initial_state = None
        time_major=False                # False:(batch,time,step,input);
                                        # True:(time,batch,step,input)
        )
output = tf.layers.dense(outputs[:, -1, :], 10)         # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)   # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(                 # return (acc,update_op)
        labels=tf.argmax(tf_y, axis=1), 
        predictions=tf.argmax(output, axis=1)
        )[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

for step in range(1200):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([train_op, loss], {tf_x:b_x, tf_y:b_y})
    if step % 50 == 0:
        accuracy_ = sess.run(accuracy, {tf_x:b_x, tf_y:b_y})
        print 'Iteration:', step, 'train loss:%.4f' %loss_, '| test accruacy:%.2f' % accuracy_

# print predictions
test_output = sess.run(output, {tf_x: test_x[:100]})
pred_y = np.argmax(test_output, 1)
print 'prediction number', pred_y
print 'real number', np.argmax(test_y[:100], 1)





