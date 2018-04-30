#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:56 2018

@author: hanxy
"""
import gym
from rl_brain import DeepQNetwork

env = gym.make('MountainCar-v0')
env = env.unwrapped

print env.action_space
print env.observation_space
print env.observation_space.high
print env.observation_space.low

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01,
                  e_greedy=0.9,
                  replace_target_iter=300,
                  memory_size=3000,
                  e_greedy_increment=0.002)

total_steps = 0

for i_episode in range(100):

    observation = env.reset()
    ep_r = 0
    while True:
        env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        position, velocity = observation_

        # the higher the better

        reward = abs(position - (-0.5))

        RL.store_transition(observation, action, reward, observation_)

        if total_steps > 1000:
            RL.learn()

        ep_r += reward

        if done:
            get = '|Get' if observation_[0] >= env.unwrapped.goal_position else '| ----'
            print 'Episode: %d %s| ep_r: %f | epsilon: %f' % (i_episode, get, round(ep_r, 2), round(RL.epsilon, 2))
            break

        observation = observation_
        total_steps += 1

RL.plot_cost()
