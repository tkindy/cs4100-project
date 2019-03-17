#!/usr/bin/python3

import gym
import sys
import time

env = gym.make('Centipede-v0')
screensDir = sys.argv[1]

for episode_num in range(100):
  observation = env.reset()
  observation.dump("{}/{}-{}-start.npy".format(screensDir, time.time(), episode_num))

  for t in range(10000):
    # env.render()
    action = env.action_space.sample()
    observation, _, done, _ = env.step(action)

    observation.dump("{}/{}-{}-{}.npy".format(screensDir, time.time(), episode_num, t))

    if done:
      print("Episode {} finished after {} timesteps".format(episode_num, t + 1))
      break
