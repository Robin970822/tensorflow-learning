#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 21:10:42 2018

@author: hanxy
"""

from maze_env import Maze
from rl_brain import QLearningTable


def update():
    for episode in range(100):
        print "Episode %d" %(episode)
#        time.sleep(1)
        # initial observation
        observation = env.reset()
        
        while True:
            # fresh env
            env.render()
            
            action = RL.choose_action(str(observation))
#            print action
#            time.sleep(1)
            
            observation_, reward, done = env.step(action)
            
            RL.learn(str(observation), action, reward, str(observation_))
            
            observation = observation_
            
            if done:
                break
    
    print 'Game Over'
    print RL.q_table
    env.destroy()
            

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    
    env.after(100, update)
    env.mainloop()
