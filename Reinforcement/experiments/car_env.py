# coding=utf-8
import pyglet
import numpy as np


class CarViewer(pyglet.window.Window):

    def __init__(self, car, goal):
        super(CarViewer, self).__init__(width=400, height=400,
                                        resizable=False, caption='car', vsync=False)

        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.car_info = car
        self.goal_info = goal
        self.center_coord = np.array([200, 200])
        self.batch = pyglet.graphics.Batch()
        self._build_env()

    def render(self):
        self._update_goal()
        self._update_car()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.goal_info['x'] = x
        self.goal_info['y'] = y

    def _build_ground(self):
        # add goal
        self.goal = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', self._set_goal()
             ),
            ('c3B', (86, 109, 249) * 4))

        # add car
        self.car = self.batch.add(
            4, pyglet.gl.GL_QUADS, None,
            ('v2f', self._set_car()
             ),
            ('c3B', (249, 86, 86) * 4)
        )

    def _update_goal(self):
        # update goal
        self.goal.vertices = self._set_goal()

    def _update_car(self):
        # update car
        self.car.vertices = self._set_car()

    def _set_goal(self):
        return [self.goal_info['x'] - self.goal_info['r'],
                self.goal_info['y'] - self.goal_info['r'],
                self.goal_info['x'] - self.goal_info['r'],
                self.goal_info['y'] + self.goal_info['r'],
                self.goal_info['x'] + self.goal_info['r'],
                self.goal_info['y'] + self.goal_info['r'],
                self.goal_info['x'] + self.goal_info['r'],
                self.goal_info['y'] - self.goal_info['r'],
                ]

    def _set_car(self):
        return [self.car_info['x'] - self.car_info['r'],
                self.car_info['y'] - self.car_info['r'],
                self.car_info['x'] - self.car_info['r'],
                self.car_info['y'] + self.car_info['r'],
                self.car_info['x'] + self.car_info['r'],
                self.car_info['y'] + self.car_info['r'],
                self.car_info['x'] + self.car_info['r'],
                self.car_info['y'] - self.car_info['r'],
                ]


class CarEnv(object):
    viewer = None
    dt = 0.1
    action_bound = [-1, 1]
    goal = {'x': 100, 'y': 100, 'l': 40}
    state_dim = 9

    def __init__(self):
        pass
