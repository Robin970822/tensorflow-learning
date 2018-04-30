#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 09:38:41 2018

@author: hanxy
"""

import tensorflow as tf

tf.set_random_seed(1)

with tf.name_scope("a_name_scope"):
    initializer = tf.constant_initializer(value=1)
    var1 = tf.get_variable(name='var1', shape=[1],
                           dtype=tf.float32, initializer=initializer)
    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)
    var21 = tf.Variable(name='var2', initial_value=[2, 1], dtype=tf.float32)

with tf.variable_scope("a_variable_scope") as scope:
    initializer = tf.constant_initializer(value=3)
    var3 = tf.get_variable(name='var3', shape=[1],
                           dtype=tf.float32, initializer=initializer)
    var4 = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)
    var4_reuse = tf.Variable(name='var4', initial_value=[4], dtype=tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print var1.name, sess.run(var1)
    print var2.name, sess.run(var2)
    print var21.name, sess.run(var21)
    print var3.name, sess.run(var3)
    print var4.name, sess.run(var4)
    print var4_reuse.name, sess.run(var4_reuse)
