#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:29:18 2018

@author: hanxy
"""
import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels
SQUAR = UNIT*3/8
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 6, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - SQUAR, hell1_center[1] - SQUAR,
            hell1_center[0] + SQUAR, hell1_center[1] + SQUAR,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 6])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - SQUAR, hell2_center[1] - SQUAR,
            hell2_center[0] + SQUAR, hell2_center[1] + SQUAR,
            fill='black')
        # hell
        hell3_center = origin + np.array([UNIT * 5, UNIT * 6])
        self.hell3 = self.canvas.create_rectangle(
            hell3_center[0] - SQUAR, hell3_center[1] - SQUAR,
            hell3_center[0] + SQUAR, hell3_center[1] + SQUAR,
            fill='black')
        
        # hell
        hell4_center = origin + np.array([UNIT * 6, UNIT * 5])
        self.hell4 = self.canvas.create_rectangle(
            hell4_center[0] - SQUAR, hell4_center[1] - SQUAR,
            hell4_center[0] + SQUAR, hell4_center[1] + SQUAR,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 7
        self.oval = self.canvas.create_oval(
            oval_center[0] - SQUAR, oval_center[1] - SQUAR,
            oval_center[0] + SQUAR, oval_center[1] + SQUAR,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - SQUAR, origin[1] - SQUAR,
            origin[0] + SQUAR, origin[1] + SQUAR,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - SQUAR, origin[1] - SQUAR,
            origin[0] + SQUAR, origin[1] + SQUAR,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 100
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2),
                    self.canvas.coords(self.hell3), self.canvas.coords(self.hell4)]:
            reward = -10
            done = True
            s_ = 'terminal'
        else:
            reward = -1
            done = False

        return s_, reward, done

    def render(self):
        time.sleep(0.1)
        self.update()


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
