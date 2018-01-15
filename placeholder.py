#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:58:09 2017

@author: hanxy
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess :
    print sess.run(output, feed_dict={input1:[7.], input2:[2.]})