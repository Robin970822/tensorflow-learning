#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:56 2018

@author: hanxy
"""
import gym
from rl_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        double_q=False,
        sess=sess
    )
with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE,
        n_features=3,
        memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001,
        double_q=True,
        sess=sess,
        output_graph=True)

sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    observation = env.reset()
    while True:
        env.render()
        action = RL.choose_action(observation)

        # convert to [-2,2] float actions
        f_action = (action - (ACTION_SPACE - 1) / 2) / ((ACTION_SPACE - 1) / 4)
        observation_, reward, done, info = env.step(np.array([f_action]))

        # normalize to (-1, 0)
        reward /= 10

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > MEMORY_SIZE:
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:
            break

        observation = observation_
        total_steps += 1
    return RL.q


print 'Natural DQN'
q_natural = train(natural_DQN)

print 'Double DQN'
q_double = train(double_DQN)

env.close()

plt.plot(np.array(q_natural), c='r', label='natural')
plt.plot(np.array(q_double), c='b', label='double')
plt.legend(loc='best')
plt.ylabel('Q eval')
plt.xlabel('Training steps')
plt.grid()
plt.show()
