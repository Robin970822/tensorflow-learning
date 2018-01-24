#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:04:44 2018

@author: hanxy
"""
import tensorflow as tf
import numpy as np

GAMMA = 0.9;

class Critic(object):
    
    def __init__(self, sess, n_features, learning_rate=0.01):
        self.sess = sess
        
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, "r")
        
        with tf.variable_scope("Critic"):
            l1 = tf.layers.dense(
                    inputs=self.s,
                    units=20,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name="l1"
                    )
            self.v = tf.layers.dense(
                    inputs=l1,
                    units=1,
                    activation=None,
                    kernel_initializer=tf.random_normal_initializer(0, 0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name="V"
                    )
        with tf.variable_scope("squared_TD_error"):
            self.td_err = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_err)
            
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        
    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        
        v_ = self.sess.run(self.v, feed_dict={self.s: s_})
        
        td_error, _ = self.sess.run([self.td_err, self.train_op], 
                                    feed_dict={self.s: s,
                                               self.v_: v_,
                                               self.r: r})
        return td_error