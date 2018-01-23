#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:29:58 2018

@author: hanxy
"""

import tensorflow as tf
import numpy as np

class PolicyGradient:
    
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.95, output_graph=False):
        self.n_actions = n_actions
        self.n_features = n_features
        self.LR = learning_rate
        self.GAMMA = reward_decay
        
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        
        self._build_net()
        
        self.sess = tf.Session()
        
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)
            
        self.sess.run(tf.global_variables_initializer())
    
    def _build_net(self):
        with tf.name_scope("inputs"):
            # 接收observation
            self.tf_obs = tf.placeholder(tf.float32, [None, self.n_features], name="observation")
            # 接收我们在这个回合中选过的actions
            self.tf_acts = tf.placeholder(tf.int32, [None,], name="actions_num")
            # 接收每一个state-action对应的value
            self.tf_vt = tf.placeholder(tf.float32, [None,], name="actions_value")
        # fc1
        layer = tf.layers.dense(
                inputs=self.tf_obs,
                units=10,
                activation=tf.nn.tanh,
                kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc1'
                )
        # fc2
        all_act = tf.layers.dense(
                inputs=layer,
                units=self.n_actions,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0, stddev=0.1),
                bias_initializer=tf.constant_initializer(0.1),
                name='fc2'
                )
        # 激励函数softmax
        self.all_act_prob = tf.nn.softmax(all_act, name = 'act_prob')
        
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=all_act, labels=self.tf_acts)
            loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.LR).minimize(loss)
    
    def choose_action(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, 
                                     feed_dict={self.tf_obs: observation[np.newaxis,:]
                                             })
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
        return action
        
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)
    
    def learn(self):
        # 衰减，并标准化这回合的reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        
        # train on episode
        self.sess.run(self.train_op, 
                      feed_dict={
                              self.tf_obs: np.vstack(self.ep_obs),
                              self.tf_acts: np.array(self.ep_as),
                              self.tf_vt: discounted_ep_rs_norm})
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]
        return discounted_ep_rs_norm
    
    def _discount_and_norm_rewards(self):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add*self.GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
            
        # normalize episode rewards
        mean = np.mean(discounted_ep_rs)
        std = np.std(discounted_ep_rs)
        discounted_ep_rs = (discounted_ep_rs - mean)/std
        return discounted_ep_rs
    