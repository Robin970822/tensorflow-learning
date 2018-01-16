#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 22:14:29 2018

@author: hanxy
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:26:10 2018

@author: hanxy
"""
import tensorflow as tf
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST', one_hot=False)

# Visualize decoder setting
# Parameters
LR = 0.001
training_epoch = 40
batch_size = 256
display_step = 1

# Network Parameters
n_input = 784 #MNIST data input (28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# hidden layer settings
n_hidden_1 = 256    # 1st layer num features
n_hidden_2 = 64     # 2nd layer num features
n_hidden_3 = 10
n_hidden_4 = 2

weights = {
        'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        
        'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
        'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
        'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        }
bias = {
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
        
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
        'decoder_b4': tf.Variable(tf.random_normal([n_input]))
        }
# Build the encoder
def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   bias['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   bias['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   bias['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   bias['encoder_b4'])
    return layer_4

# Build the decoder
def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   bias['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   bias['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   bias['decoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   bias['decoder_b4'])
    return layer_4

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Labels are the input data
y_true = X

# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(LR).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Lauch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)
    # Training cycle
    for epoch in range(training_epoch):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run 
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        print "Iteration: %04d " %(epoch), "cost=","{:.9f}".format(c)
    print "Optimization Finished!"
    
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    plt.scatter(encoder_result[:,0], encoder_result[:,1], c=mnist.test.labels)
    plt.show()