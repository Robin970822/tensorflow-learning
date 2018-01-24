#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 16:40:15 2018

@author: hanxy
"""

import tensorflow as tf
import numpy as np
import gym
from actor import Actor
from critic import Critic

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODE = 3000
DISPLAY_THRESHOLD = 200
MAX_EP_STEP = 1000
RENDER = False

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

sess = tf.Session()

actor = Actor(sess,
              n_actions=env.action_space.n, 
              n_features=env.observation_space.shape[0], 
              learning_rate=0.001)

critic = Critic(sess,
                n_features=env.observation_space.shape[0],
                learning_rate=0.01)

sess.run(tf.global_variables_initializer())

tf.summary.FileWriter("logs/", sess.graph)

for i_episode in range(MAX_EPISODE):
    observation = env.reset()
    t = 0
    track_reward = []
    while True:
        if RENDER: env.render()
        
        action = actor.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        
        if done: reward = -20
        
        track_reward.append(reward)
        
        td_error = critic.learn(observation, reward, observation_)
        actor.learn(observation, action, td_error)
        
        observation = observation_
        t += 1
        
        if done or t > MAX_EP_STEP:
            ep_rs_sum = sum(track_reward)
            
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum *0.05
            if running_reward > DISPLAY_THRESHOLD: RENDER = True
            print "Episode: %d | Reward: %d" %(i_episode, running_reward)
            break
        
        