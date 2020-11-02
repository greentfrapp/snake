from stable_baselines3 import DQN
from stable_baselines3.common.cmd_util import make_vec_env

from env import SnakeEnv
from learn import CustomCNN


# Train the agent
policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
)

# Instantiate the env
dim = 7
env = SnakeEnv(dim)
# wrap it
env = make_vec_env(lambda: env, n_envs=1)

model = DQN.load(f"snake_{dim}x{dim}_2")

for i in range(10):
    obs = env.reset()
    n_steps = 500
    total_reward = 0
    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        # print(action)
        obs, reward, done, info = env.step(action)
        # env.render()
        total_reward += reward
        if done:
            print(f"Agent died at step {step}! Total reward={total_reward}")
            break
    if not done:
        print(f"Agent survived after {n_steps} steps! Total reward={total_reward}")
