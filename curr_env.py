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
N_CHANNELS = 3  # food, own-head, own-body


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, width=32, height=None, curriculum=None):
        super(SnakeEnv, self).__init__()
        self.width = width
        self.height = height or width
        self.curriculum = curriculum or width
        
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=
                        (self.height, self.width, N_CHANNELS), dtype=np.uint8)

    def step(self, action):
        done = False
        reward = 0  # -0.1
        state = np.zeros((self.height, self.width, N_CHANNELS))
        not_empty = np.zeros((self.height, self.width))

        if action == UP:
            change = np.array([-1, 0])
        if action == DOWN:
            change = np.array([1, 0])
        if action == LEFT:
            change = np.array([0, -1])
        if action == RIGHT:
            change = np.array([0, 1])

        head = self.own_pos[0] + change
        if (head == self.own_pos[1]).all(0):
            reward = -1
            done = True
        if (self.own_pos[:-1] == head).all(1).any():
            reward = -1
            done = True
        if head[0] < 0 or head[0] >= self.height or \
                head[1] < 0 or head[1] >= self.width:
            reward = -1
            done = True
            head = self.own_pos[0]
        self.own_pos = np.insert(self.own_pos, 0, head, axis=0)
        if not np.array_equal(self.food_pos, head):
            self.own_pos = self.own_pos[:-1]

        state[head[0], head[1], 1] = 255  # Own head
        for px in self.own_pos:
            state[px[0], px[1], 2] = 255  # Own body inc. head
            not_empty[px[0], px[1]] = 255
        for i in range(self.curriculum, self.width):
            for j in range(self.width):
                not_empty[i, j] = 255
                not_empty[j, i] = 255

        if np.array_equal(self.food_pos, head):
            reward = 10
            empty_space = np.concatenate([np.expand_dims(d, 1) for d in (not_empty == 0).nonzero()], axis=1)
            if len(empty_space) == 0:
                reward = 100
                done = True
                return state, reward, done, {}
            else:
                self.food_pos = empty_space[np.random.choice(len(empty_space))]

        state[self.food_pos[0], self.food_pos[1], 0] = 255  # Food
        return state, reward, done, {}

    def reset(self):
        state = np.zeros((self.height, self.width, N_CHANNELS))
        not_empty = np.zeros((self.height, self.width))
        self.own_pos = np.array([
            [1, 0], # Head
            [0, 0],
        ])
        state[self.own_pos[0, 0], self.own_pos[0, 1], 1] = 255  # Own head
        for px in self.own_pos:
            state[px[0], px[1], 2] = 255  # Own body inc. head
            not_empty[px[0], px[1]] = 255
        for i in range(self.curriculum, self.width):
            for j in range(self.width):
                not_empty[i, j] = 255
                not_empty[j, i] = 255
        empty_space = np.concatenate([np.expand_dims(d, 1) for d in (not_empty == 0).nonzero()], axis=1)
        self.food_pos = empty_space[np.random.choice(len(empty_space))]
        state[self.food_pos[0], self.food_pos[1], 0] = 255  # Food
        return state

    def render(self, mode='console', close=False):
        # Render the environment to the screen
        render = np.zeros((self.height, self.width))
        render[self.food_pos[0], self.food_pos[1]] = 1
        head = self.own_pos[0]
        render[head[0], head[1]] = 2
        for px in self.own_pos[1:]:
            render[px[0], px[1]] = 3
        for i in range(self.curriculum, self.width):
            for j in range(self.width):
                render[i, j] = 9
                render[j, i] = 9
        print(render)
        return render


if __name__ == "__main__":
    env = SnakeEnv(6, curriculum=4)
    env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()
        action = input("Action: ")
        if action == 'w':
            action = 0
        if action == 's':
            action = 1
        if action == 'a':
            action = 2
        if action == 'd':
            action = 3
        _, reward, done, _ = env.step(action)
        print(reward)
        total_reward += reward
    print(f"Total reward: {total_reward}")