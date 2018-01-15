#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 19:49:39 2017

@author: hanxy
"""

import tensorflow as tf

state = tf.Variable(0, name='counter')
#print state.name
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.initialize_all_variables() # must have if define variable

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print sess.run(update)
        print sess.run(state)