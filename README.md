# snake
Simple Snake environment for reinforcement learning.

## Topics

- Effects of environment parameters
	- Grid size
	- Per step reward
	- Death reward
	- Food reward
- Curriculum Learning
- Adversarial Snake

## Experiments

- Grid sizes
	- 4, 6, 8, 10, ..., 20
	- Curriculum learning for 20

# Writeup

## Introduction

- Reinforcement Learning
- Curriculum Learning

Reinforcement learning revolves around the notion of an agent learning to take a sequence of actions that maximizes the total reward obtained in a given environment. The environment is often modeled as a Markov decision process, comprising of a state space, a transition matrix, an action space and a set of reward mappings. The state space consists of all the possible states that the environment can take on. The transition matrix refers models the distributions of the next state $$s'$$, given the current state $$s$$ and an action $$a$$. The action space consists of all the possible actions. The reward mappings refer to the immediate reward that the agent receives from taking action $$a$$ in state $$s$$.

While previous reinforcement learning algorithms have relied on handcrafted features to model the environment, recent works have moved towards the use of deep learning methods. These methods rely on the use of general learners, typically neural networks and variants, in order to extract high-level features from a given state. Despite challenges such as sparse and noisy rewards or labels and highly correlated samples in the training set, deep learning methods have demonstrated superior performance. In many cases, these methods have also resulted in superhuman agents that have beaten human world champions in very complex games such as Go, Dota 2 and Starcraft 2.

The sparsity of rewards in reinforcement learning has been a major challenge impeding effective implementation of learning algorithms. Specifically, these algorithms typically begin with a randomly initialized agent. This untrained agent explores the environment and learns from rollouts of its exploration experiences. However, if the environment is too difficult or have very few reward signals when explored randomly, the agent is unable to effectively learn from its experiences.

Consider the game of Snake, where an agent-controlled snake moves around in a grid world and gains points from eating food pellets. If the grid world is too large, an untrained agent is unlikely to randomly chance upon a single food pellet. As such, the agent might only learn to avoid death but never learns to move towards food pellets. Alternatively, the agent takes an extremely long time to learn from a slow accumulation of these rare instances where it randomly eats up a pellet.

One approach to alleviate this is the concept of curriculum learning. As its name suggests, curriculum learning uses a "curriculum" - a modification to the environment such that initial levels are easier, with more common rewards, and gradually increasing in difficulty as the agent progresses.

This project explores the use of curriculum learning on the game of Snake. More precisely, we first examine the performance of a DQN algorithm on Snake environments with varying grid sizes, ranging from 4 by 4 to 20 by 20. Then, we demonstrate the effectiveness of curriculum learning on a 20 by 20 Snake environment and show that the agent's improvement is vastly accelerated compared to the baseline.

## Problem

- Snake

Snake is a classic video game where the player typically controls a snake in a 2D grid world. The snake is simply a line of pixels. Points are scored whenever the snake comes into contact with food pellets, which are commonly represented as single pixels. In addition, the snake also increases in length when it consumes food pellets. The game ends when the snake touches or "eats" itself.

For this project, we implemented a custom version of Snake that inherits OpenAI gym's gym.Env class. This helped to integrate our work with various open-source libraries such as OpenAI baselines and stable-baselines.

In our custom implementation, the snake starts off as a line of 2 pixels and gains a pixel whenever it consumes a pellet. In addition to the rules described above, the game also ends when the snake exceeds the boundary of the grid world. The reward scheme is as follows:

- Consuming a pellet: +10
- Eating itself or exceeding boundary: -1 and game ends
- Consuming the last pellet (i.e. no free space left on grid): +100 and game ends
- Otherwise: 0

Finally, to facilitate our experiments, our custom implementation also allows for defining of the grid world dimensions (i.e. height and width).

## Related Work

- DQN
- Curriculum Learning

## Experiments

- DQN Details
- Grid sizes results (naive)
- Curriculum results
	- 411 at 2.5e6 iterations

## Conclusion
