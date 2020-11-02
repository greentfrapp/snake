"""
adv_env.py

Code for Snake environment, compatible with OpenAI Gym.
With adversary.
"""

import gym
from gym import spaces
import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
N_ACTIONS = 4
N_CHANNELS = 5  # food, own-head, own-body, adv-head, adv-body

UPDATE = {
    UP: np.array([-1, 0]),
    DOWN: np.array([1, 0]),
    LEFT: np.array([0, -1]),
    RIGHT: np.array([0, 1]),
}


class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human', 'console']}

    def __init__(self, width=32, height=None):
        super(SnakeEnv, self).__init__()
        self.width = width
        self.height = height or width

        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, N_CHANNELS),
            dtype=np.uint8
        )

    def step(self, action):
        done = False
        reward = 0  # -0.1
        state = np.zeros((self.height, self.width, N_CHANNELS))
        adv_action = self.adv_step()

        head = self.own_pos[0] + UPDATE[action]
        if np.array_equal(head, self.own_pos[1]):
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

        adv_head = self.adv_pos[0] + UPDATE[adv_action]
        if not done and np.array_equal(adv_head, self.adv_pos[1]):
            reward = 100
            done = True
        if not done and (self.adv_pos[:-1] == adv_head).all(1).any():
            reward = 100
            done = True
        if not done and adv_head[0] < 0 or adv_head[0] >= self.height or \
                adv_head[1] < 0 or adv_head[1] >= self.width:
            reward = 100
            done = True
            adv_head = self.adv_pos[0]
        self.adv_pos = np.insert(self.adv_pos, 0, adv_head, axis=0)
        if not np.array_equal(self.food_pos, adv_head):
            self.adv_pos = self.adv_pos[:-1]

        if not done and np.array_equal(head, adv_head):
            reward = -1
            done = True
        if not done and (self.own_pos[:-1] == adv_head).all(1).any():
            reward = 100
            done = True
        if not done and (self.adv_pos[:-1] == head).all(1).any():
            reward = -1
            done = True

        state[head[0], head[1], 1] = 255  # Own head
        for px in self.own_pos:
            state[px[0], px[1], 2] = 255  # Own body inc. head
        state[adv_head[0], adv_head[1], 3] = 255  # Adv head
        for px in self.adv_pos:
            state[px[0], px[1], 4] = 255  # Adv body inc. head

        if not done and np.array_equal(self.food_pos, head):
            reward = 10
            empty_space = np.concatenate([np.expand_dims(d, 1) for d in ((state[:, :, 2] + state[:, :, 4]) == 0).nonzero()], axis=1)
            if len(empty_space) == 0:
                reward = 100
                done = True
                return state, reward, done, {}
            else:
                self.food_pos = empty_space[np.random.choice(len(empty_space))]

        if not done and np.array_equal(self.food_pos, adv_head):
            # reward = 0
            empty_space = np.concatenate([np.expand_dims(d, 1) for d in ((state[:, :, 2] + state[:, :, 4]) == 0).nonzero()], axis=1)
            if len(empty_space) == 0:
                # reward = -100
                done = True
                return state, reward, done, {}
            else:
                self.food_pos = empty_space[np.random.choice(len(empty_space))]

        state[self.food_pos[0], self.food_pos[1], 0] = 255  # Food
        return state, reward, done, {}

    def reset(self):
        state = np.zeros((self.height, self.width, N_CHANNELS))
        self.own_pos = np.array([
            [1, 0],  # Head
            [0, 0],
        ])
        self.adv_pos = np.array([
            [self.height-2, self.width-1],  # Head
            [self.height-1, self.width-1],
        ])
        state[self.own_pos[0, 0], self.own_pos[0, 1], 1] = 255  # Own head
        for px in self.own_pos:
            state[px[0], px[1], 2] = 255  # Own body inc. head
        state[self.adv_pos[0, 0], self.adv_pos[0, 1], 3] = 255  # Adv head
        for px in self.adv_pos:
            state[px[0], px[1], 4] = 255  # Adv body inc. head
        empty_space = np.concatenate([np.expand_dims(d, 1) for d in ((state[:, :, 2] + state[:, :, 4]) == 0).nonzero()], axis=1)
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
        adv_head = self.adv_pos[0]
        render[adv_head[0], adv_head[1]] = 4
        for px in self.adv_pos[1:]:
            render[px[0], px[1]] = 5
        print(render)
        return render

    def adv_step(self):
        food_direction = self.food_pos - self.adv_pos[0]
        priorities = [
            [-food_direction[0], UP],
            [food_direction[0], DOWN],
            [-food_direction[1], LEFT],
            [food_direction[1], RIGHT],
        ]
        priorities.sort(reverse=True)
        next_pos = [
            [self.adv_pos[0] + np.array([-1, 0]), UP],
            [self.adv_pos[0] + np.array([1, 0]), DOWN],
            [self.adv_pos[0] + np.array([0, -1]), LEFT],
            [self.adv_pos[0] + np.array([0, 1]), RIGHT],
        ]
        free = []
        for p in next_pos:
            if not (self.own_pos == p[0]).all(1).any() and \
                    not (self.adv_pos == p[0]).all(1).any() and \
                    not (p[0][0] < 0 or p[0][0] >= self.height) and \
                    not (p[0][1] < 0 or p[0][1] >= self.width):
                free.append(p[1])
        for _, action in priorities:
            if action in free:
                return action
        return np.random.choice(N_ACTIONS)


if __name__ == "__main__":
    env = SnakeEnv(5)
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