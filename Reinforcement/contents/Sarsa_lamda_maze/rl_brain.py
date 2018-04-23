#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:45 2018

@author: hanxy
"""

import numpy as np
import pandas as pd

class RL(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.LR = learning_rate
        self.GAMMA = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
    
    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action
    
    def learn(self, *agrs):
        pass
    
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                    pd.Series(
                            [0]*len(self.actions),
                            index=self.q_table.columns,
                            name=state))


class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.GAMMA * self.q_table.loc[s_, a].max()
        else:
            q_target = r
        self.q_table.loc[s, a] += self.LR * (q_target - q_predict)


# on-policy
class SarsaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)
        
    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.GAMMA * self.q_table.loc[s_, a_]
        else:
            q_target = r
        self.q_table.loc[s, a] += self.LR * (q_target - q_predict)


# backward eligibility traces
class SarsaLambdaTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            to_be_append = pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            self.q_table = self.q_table.append(to_be_append)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(to_be_append)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.GAMMA * self.q_table.loc[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        error = q_target - q_predict

        # increase trace amount for visited state-action pair

        # Method 1:
        # self.eligibility_trace.loc[s, a] += 1

        # Method 2:
        self.eligibility_trace.loc[s, :] *= 0
        self.eligibility_trace.loc[s, a] = 1

        # Q update
        self.q_table += self.LR * error * self.eligibility_trace

        # decay eligibility trace after update
        self.eligibility_trace *= self.GAMMA * self.lambda_