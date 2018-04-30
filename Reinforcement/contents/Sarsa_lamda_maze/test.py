#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:05:56 2018

@author: hanxy
"""
from maze_env import Maze
from rl_brain import SarsaLambdaTable


def update():
    for episode in range(100):
        print "Episode %d" % (episode)
        #        time.sleep(1)
        # initial observation
        observation = env.reset()
        action = RL.choose_action(str(observation))
        RL.eligibility_trace *= 0

        while True:
            # fresh env
            env.render()

            action_ = RL.choose_action(str(observation))
            #            print action
            #            time.sleep(1)

            observation_, reward, done = env.step(action)

            RL.learn(str(observation), action, reward, str(observation_), action_)

            observation = observation_
            action = action_

            if done:
                break

    print 'Game Over'
    print RL.q_table
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()
