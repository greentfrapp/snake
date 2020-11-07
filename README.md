# Curriculum Learning with Snake

A Stanford CS238/AA228 final project.

## Abstract

Sparse rewards or a difficult environment is often a challenge for reinforcement learning algorithms that begin with randomly initialized agents. Curriculum learning, or progressively increasing the difficulty of environments, has been suggested as a way to alleviate this problem. A random agent starts off on a simple level with easy and common rewards and the environment gets increasingly harder as the agent improves. In this project, we apply curriculum learning to Snake and demonstrate how it helps to accelerate the learning of an agent, especially for large grid sizes.

## Instructions

To reproduce the results in the paper, run the following:

```
# Vanilla training
$ python train_vanilla.py
```

```
# Training with curriculum
$ python train_curriculum.py
```

After running the experiments, run the following to generate the graphs:

```
$ python plot.py
```

## Environments

Environments are found in `env.py`.

### Vanilla Environment `SnakeEnv`

In this implementation of Snake, the snake starts off as a line of 2 pixels and gains a pixel whenever it consumes a pellet. The game ends when the snake touches itself or exceeds the boundary of the grid world. The reward scheme is as follows:

- Consuming a pellet: `+10`
- Eating itself or exceeding boundary: `-1` and game ends
- Consuming the last pellet possible: `+100` and game ends
- Otherwise: `0`

The environment can be run as follows:

```
env = SnakeEnv(size) # size determines the height and width of the grid

obs = env.reset()
while True:
	# 4 available actions corresponding to UP, DOWN, LEFT, RIGHT
	action = np.random.choice(4)
	obs, reward, done, info = env.step(action)
	if done: break
```

### Curriculum Environment `CurrSnakeEnv`

This environment is similar to the vanilla environment except it takes a `curriculum` parameter that ranges from 0 to 1 inclusive.

Instead of randomly placing a single food pellet, we start by placing food pellets on randomly selected `curriculum * 100%` of the grid world. Only the last pellet is replenished to ensure at least one pellet remains in the grid world.

Also, instead of immediately ending the game when the snake touches itself or exceeds the boundary of the grid world, there is a `curriculum * 100%` chance that the agent continues the game from the previous state, with a reward of `-1`.

Initialize the environment with `env = CurrSnakeEnv(size, curriculum)`.

## Agent

We train an agent with the DQN algorithm implemented in `stable_baselines3` along with a custom policy network that can be found in `model.py`.

## GUI

After training an agent, you can visualize the performance of the agent by running a GUI with the following commands in two Terminal windows.

```
$ python -m http.server 3000
```

```
uvicorn app:app
```

Then navigate to `http://localhost:3000` with your favorite browser!
