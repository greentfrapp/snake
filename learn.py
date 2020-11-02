from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym

import torch as th
import torch.nn as nn
import numpy as np

from env import SnakeEnv


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


def play(env, model, n_steps=500):
    obs = env.reset()
    total_reward = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Goal reached at step {step}! Total reward={total_reward}")
            break
    if not done:
        print(f"Agent survived after {n_steps} steps! Total reward={total_reward}")
    return total_reward


if __name__ == "__main__":
    # Train the agent
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )

    # Instantiate the env
    size = 7
    env = SnakeEnv(size)
    # wrap it
    env = make_vec_env(lambda: env, n_envs=1)

    # Train the agent
    model = DQN('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
    # model.load("snake_4x4")

    highest_reward = -100
    i = 1
    while True:
        model.learn(10000, reset_num_timesteps=False, log_interval=1000)
        print(f"Testing agent after {i * 10000} iterations...")
        scores = []
        for n in range(10):
            scores.append(play(env, model))
        mean_score = np.mean(scores)
        if mean_score >= highest_reward:
            highest_reward = mean_score
            model.save(f"snake_{size}x{size}_2")
            print(f"Best agent trained with avg. score of {mean_score}, saving model...")
        else:
            print(f"Avg. score {mean_score} is lower than best avg. score of {highest_reward}.")
        i += 1
