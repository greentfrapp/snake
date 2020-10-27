"""
env.py

Code for Snake environment, compatible with OpenAI Gym.
"""

import gym
from gym import spaces
import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
N_ACTIONS = 4
N_CHANNELS = 3 # food, own-head, own-body


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, width=32, height=None):
        super(SnakeEnv, self).__init__()
        self.width = width
        self.height = height or width
        
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=1, shape=
                        (N_CHANNELS, self.height, self.width), dtype=np.float32)

    def step(self, action):
        done = False
        reward = 0
        state = np.zeros((N_CHANNELS, self.height, self.width))

        if action == UP:
            change = np.array([-1, 0])
        if action == DOWN:
            change = np.array([1, 0])
        if action == LEFT:
            change = np.array([0, -1])
        if action == RIGHT:
            change = np.array([0, 1])

        head = self.own_pos[0] + change
        if (self.own_pos == head).all(1).any():
            reward = -10
            done = True
        if head[0] < 0 or head[0] >= self.height or \
            head[1] < 0 or head[1] >= self.width:
            reward = -10
            done = True
            head = self.own_pos[0]
        self.own_pos = np.insert(self.own_pos, 0, head, axis=0)[:-1]
        state[1, self.own_pos[0, 0], self.own_pos[0, 1]] = 1 # Own head
        for px in self.own_pos:
            state[2, px[0], px[1]] = 1 # Own body inc. head
        
        if np.array_equal(self.food_pos, head):
            reward = 1
            empty_space = np.concatenate([np.expand_dims(d, 1) for d in (state[2] == 0).nonzero()], axis=1)
            self.food_pos = np.random.choice(empty_space)
        state[0, self.food_pos[0], self.food_pos[1]] = 1 # Food

        return state, reward, done, {}

    def reset(self):
        state = np.zeros((N_CHANNELS, self.height, self.width))
        self.own_pos = np.array([
            [1, 0], # Head
            [0, 0],
        ])
        state[1, self.own_pos[0, 0], self.own_pos[0, 1]] = 1 # Own head
        for px in self.own_pos:
            state[2, px[0], px[1]] = 1 # Own body inc. head
        empty_space = np.concatenate([np.expand_dims(d, 1) for d in (state[2] == 0).nonzero()], axis=1)
        self.food_pos = empty_space[np.random.choice(len(empty_space))]
        state[0, self.food_pos[0], self.food_pos[1]] = 1 # Food
        return state

    def render(self, mode='console', close=False):
        # Render the environment to the screen
        render = np.zeros((self.height, self.width))
        render[self.food_pos[0], self.food_pos[1]] = 1
        head = self.own_pos[0]
        render[head[0], head[1]] = 2
        for px in self.own_pos[1:]:
            render[px[0], px[1]] = 3
        print(render)


if __name__ == "__main__":
    env = SnakeEnv(10)
    env.reset()
    done = False
    while not done:
        env.render()
        _, _, done, _ = env.step(1)
