#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:01:01 2018

@author: hanxy
"""
import tensorflow as tf
import numpy as np


class Actor(object):
    def __init__(self, sess, n_actions, n_features, learning_rate=0.001):
        self.sess = sess
        
        self.s = tf.placeholder(tf.float32, [1, n_features], name="state")
        self.a = tf.placeholder(tf.int32, None, name="action")
        self.td_error = tf.placeholder(tf.float32, None, name="TD_ERROR")
        
        with tf.variable_scope("Actor"):
            l1 = tf.layers.dense(
                    inputs=self.s,
                    units=30,
                    activation=tf.nn.relu,
                    kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name="l1"
                    )
            
            self.acts_prob = tf.layers.dense(
                    inputs=l1,
                    units=n_actions,
                    activation=tf.nn.softmax,
                    kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                    bias_initializer=tf.constant_initializer(0.1),
                    name="acts_prb"
                    )
        
        with tf.variable_scope("exp_v"):
            log_prob = tf.log(self.acts_prob[0, self.a])
            self.exp_v = tf.reduce_mean(log_prob*self.td_error)
            
        with tf.variable_scope("train"):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.exp_v)
    
    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        _, exp_v = self.sess.run([self.train_op, self.exp_v],
                                 feed_dict={self.s: s,
                                            self.a: a,
                                            self.td_error: td})
        return exp_v
    
    def choose_action(self, s):
        s = s[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, feed_dict={self.s: s})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())
