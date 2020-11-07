from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gym

import torch as th
import torch.nn as nn
import numpy as np
from pathlib import Path

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
    logs_folder = Path("logs")
    logs_folder.mkdir(parents=True, exist_ok=True)
    models_folder = Path("models")
    models_folder.mkdir(parents=True, exist_ok=True)

    for size in range(4, 11):
        for n in range(5):

            log_file = open(logs_folder / f"log_S_{size}_{n}.csv", 'w')
            log_file.write("iteration,avg score,max avg,max single\n")

            # Instantiate the env
            # size = 7
            env = SnakeEnv(size)
            # wrap it
            env = make_vec_env(lambda: env, n_envs=1)

            # Train the agent
            model = DQN('CnnPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
            # model.load("snake_4x4")

            highest_reward = -100
            highest_single = -100
            i = 1
            plateau = 0
            while True:
                model.learn(10000, reset_num_timesteps=False, log_interval=1000)
                print(f"S={size}: Testing agent after {i * 10000} iterations...")
                scores = []
                for _ in range(10):
                    scores.append(play(env, model))
                mean_score = np.mean(scores)
                highest_single = max(highest_single, float(max(scores)))
                log_file.write(f"{i*10000},{mean_score},{max(mean_score, highest_reward)},{highest_single}\n")
                if np.max(scores) == 10 * size ** 2 + 70:
                    model.save(models_folder / f"model_S_{size}_{n}_last")
                    print("Max score achieved, saving model...")
                    break
                elif mean_score >= highest_reward:
                    plateau = 0
                    highest_reward = mean_score
                    model.save(models_folder / f"model_S_{size}_{n}")
                    print(f"Best agent trained with avg. score of {mean_score}, saving model...")
                else:
                    plateau += 1
                    print(f"Avg. score {mean_score} is lower than best avg. score of {highest_reward}.")
                    if plateau >= 100:
                        model.save(models_folder / f"model_S_{size}_{n}_last")
                        print("Score plateaued, saving model...")
                        break
                i += 1

            log_file.close()
