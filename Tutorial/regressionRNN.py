#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 16:22:34 2018

@author: hanxy
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
BATCH_START_TEST = 0


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape(50 batch, 20 steps)
    xs = np.arange(BATCH_START,
                   BATCH_START + TIME_STEPS * BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b-')
    plt.show()
    # return seq, res and shape (batch,step,input)
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]


class LSTMRNN(object):
    def _init_(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):
        # (batch*n_step,in_size)
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')
        # Ws(in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs(cell_size)
        bs_in = self._bias_variable([self.cell_size, ])
        # l_in_y = (batch*n_steps,cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y,
                                 [-1, self.n_steps, self.cell_size],
                                 name='2_3D')

    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size,
                                                        dtype=tf.float32)
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell,
            self.l_in_y,
            initial_state=self.cell_init_state,
            time_major=False)

    def add_output_layer(self):
        # (batch*n_step,cell_size)
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size],
                             name='2_2D')
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size])
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out

    def compute_cost(self):
        pass

    def msr_error(self, y_pre, y_target):
        return tf.square(tf.sub(y_pre, y_target))

    def _weight_variable(self, shape, name='weights'):
        init = tf.random_normal_initializer(mean=0., stddev=0.1)
        return tf.get_variable(shape=shape, initializer=init, name=name)

    def _bias_variable(self, shape, name='biase'):
        init = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=init)
