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
        self.batch = pyglet.graphics.Batch()
        self._build_ground()

    def render(self):
        self._update_goal()
        self._update_car()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.goal_info['x'] = x
        self.goal_info['y'] = y

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.W:
            self.car_info['y'] += 10
        if symbol == pyglet.window.key.A:
            self.car_info['x'] -= 10
        if symbol == pyglet.window.key.S:
            self.car_info['y'] -= 10
        if symbol == pyglet.window.key.D:
            self.car_info['x'] += 10

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

    def __init__(self):
        self.viewer = None
        self.dt = 0.1

        self.action_bound = [-20, 20]
        self.state_dim = 7
        self.action_dim = 2

        self.goal = {'x': 100, 'y': 100, 'r': 40}
        self.car = {'x': 200, 'y': 200, 'r': 10}
        self.on_goal = 1. if self._on_goal() else 0.

    def step(self, action):
        done = False
        # print action
        action = np.clip(action, a_min=self.action_bound[0], a_max=self.action_bound[1])
        # print action
        car_x = self.car['x'] + action[0] * self.dt
        car_y = self.car['y'] + action[1] * self.dt

        if (0 < car_x < 400) and (0 < car_y < 400):
            self.car['x'] = car_x
            self.car['y'] = car_y

        dist = [(self.goal['x'] - self.car['x']) / 400,
                (self.goal['y'] - self.car['y']) / 400]
        r = -np.sqrt(dist[0] ** 2 + dist[1] ** 2)

        # done and reward
        if self._on_goal():
            r += 1.
            self.on_goal += 1
            done = [True if self.on_goal > 100 else False]
        else:
            self.on_goal *= 0

        # state
        s = np.concatenate(([self.car['x'], self.car['y']], [self.goal['x'], self.goal['y']], dist,
                           [1. if self.on_goal else 0.]))
        # print s
        return s, r, done

    def reset(self):
        self.goal['x'] = np.random.rand() * 400.
        self.goal['y'] = np.random.rand() * 400.
        self.car['x'] = np.random.rand()*400
        self.car['y'] = np.random.rand()*400
        # self.goal = {'x': 100, 'y': 100, 'r': 40}
        # self.car = {'x': 150, 'y': 200, 'r': 10}
        self.on_goal = 1. if self._on_goal() else 0.
        dist = [(self.goal['x'] - self.car['x']) / 400,
                (self.goal['y'] - self.car['y']) / 400]
        s = np.concatenate(([self.car['x'], self.car['y']], [self.goal['x'], self.goal['y']], dist,
                           [1. if self.on_goal else 0.]))
        return s

    def render(self):
        if self.viewer is None:
            self.viewer = CarViewer(self.car, self.goal)
        self.viewer.render()

    def _on_goal(self):
        return (self.goal['x'] - self.goal['r'] < self.car['x'] - self.car['r']) and \
               (self.car['x'] + self.car['r'] < self.goal['x'] + self.goal['r']) and \
               (self.goal['y'] - self.goal['r'] < self.car['y'] - self.car['r']) and \
               (self.car['y'] + self.car['r'] < self.goal['y'] + self.goal['r'])

    @staticmethod
    def sample_action():
        return (np.random.rand(2) - 0.5) * 10


if __name__ == "__main__":
    env = CarEnv()
    while True:
        env.render()
        # env.step(env.sample_action())
