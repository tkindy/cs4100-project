#!/usr/bin/python3

import gym

env = gym.make('Centipede-v0')

for episode_num in range(20):
  observation = env.reset()

  for t in range(10000):
    env.render()
    action = env.action_space.sample()
    observation, _, done, _ = env.step(action)

    if done:
      print("Episode {} finished after {} timesteps".format(episode_num, t + 1))
      break
