#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:56 2018

@author: hanxy
"""
import gym
from rl_brain import PrioritizedReplyDQN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

env = gym.make('Pendulum-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 10000
ACTION_SPACE = 11

sess = tf.Session()
with tf.variable_scope('natural_DQN'):
    RL_natural = PrioritizedReplyDQN(
        n_actions=11, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )

with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = PrioritizedReplyDQN(
        n_actions=11, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True, output_graph=True,
    )
sess.run(tf.global_variables_initializer())


def train(RL):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in range(20):
        print "Episode: %d" %(i_episode)
        observation = env.reset()
        while True:
            env.render()

            action = RL.choose_action(observation)
#
#            observation_, reward, done, info = env.step(action)
#
#            if done: reward = 10
        # convert to [-2,2] float actions
            f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)
            observation_, reward, done, info = env.step(np.array([f_action]))
            
            # normalize to (-1, 0)
            reward /= 10

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                RL.learn()

            if total_steps - MEMORY_SIZE > 20000:
                steps.append(total_steps)
                episodes.append(i_episode)
                break
            
            if done:
                steps.append(total_steps)
                episodes.append(i_episode)
                break

            observation = observation_
            total_steps += 1
    return np.vstack((episodes, steps))

print 'Natural DQN'
his_natural = train(RL_natural)

print 'Prioritized Reply DQN'
his_prio = train(RL_prio)

# compare based on first success
plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='r', label='DQN with prioritized replay')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
plt.grid()
plt.show()

