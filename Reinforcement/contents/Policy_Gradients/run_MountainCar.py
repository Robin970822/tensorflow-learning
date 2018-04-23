#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 21:26:20 2018

@author: hanxy
"""

import gym
from rl_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False
DISPLAY_THRESHOLD = -2000


env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(1)

print env.action_space
print env.observation_space
print env.observation_space.high
print env.observation_space.low

RL = PolicyGradient(
        n_actions=env.action_space.n,
        n_features=env.observation_space.shape[0],
        learning_rate=0.02,
        reward_decay=0.995,
        output_graph=True
        )

for i_episode in range(1000):
    
    observation = env.reset()
    
    while True:
        if RENDER: env.render()
        
        action = RL.choose_action(observation)
        
        observation_, reward, done, info = env.step(action)
        
        RL.store_transition(observation, action, reward)
        
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            
            if running_reward > DISPLAY_THRESHOLD: RENDER = True
            print "Episode: %d | Reward: %d" %(i_episode, int(running_reward))
            
            vt = RL.learn()
            
            break
        
        observation = observation_
        
plt.plot(vt)
plt.xlabel('Episode Steps')
plt.ylabel('Normalized State-Action Value')
plt.show()
