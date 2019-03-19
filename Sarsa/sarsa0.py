import gym
import math
import random
import pandas as pd
import json
import copy
from random import random
from collections import defaultdict, namedtuple
from sympy.matrices import Matrix, ImmutableMatrix
import numpy as np
import matplotlib.pyplot as plt
import sys

#An implementation of Sarsa specified for Centipede-ram-v0
class Sarsa:

  def __init__(self, numGames, displayPlot=True):
    self.env = gym.make('Centipede-ram-v0')
    self.QTable = defaultdict(float)
    self.FrequencyTable = defaultdict(int)
    self.numGames = numGames 
    self.displayPlut = displayPlot
    
  def getQTable(self):
    return self.QTable

  def getFTable(self):
    return self.FrequencyTable

  def getEnv(self):
    return self.env

  def getNumGames(self):
    return self.numGames

  def greedyPolicy(self, state, epsilon):
    randomValue = np.random.rand()
    if (randomValue < epsilon):
      return self.getEnv().action_space.sample()

    qValues = {}
    for i in range(18):
      qValues[i] = self.getQTable()[state, i]
    maxQValue = max(qValues.values())

    actions_with_max_q = [a for a, q in qValues.items() if q == maxQValue]
    return np.random.choice(actions_with_max_q)

  def updateQ(self, prevState, nextState, action, nextAction, reward, done, alpha, gamma):
    qValue = self.getQTable()[nextState, nextAction]
    self.FrequencyTable[prevState, action] = self.FrequencyTable[prevState, action] + 1
    update = alpha * (reward + gamma * qValue * (1 - done)  - self.getQTable()[prevState, action])
    self.getQTable()[prevState, action] += update

  def getNewEpsilon(self, epsilon, finalEpsilon, epsilonDrop):
    if(epsilon > finalEpsilon):
      epsilon -= epsilonDrop
      if(epsilon < finalEpsilon):
        epsilon = finalEpsilon

    return epsilon
    
  def run(self):
    print("RUN %d Games..." % self.getNumGames())
    utility, steps = 0, 0
    gamma, epsilon = 0.99, 0.99
    epsilonDecay, finalEpsilon = 0.999, 0.01
    alpha, alphaDecay = 0.6, 0.8
    allStates = namedtuple('allStates', ['state'])
    prevState = None
    prevReward = 0
    prevAction = 0
    epsilonDrop = (epsilon - finalEpsilon) / self.getNumGames() * 2
    rewards = np.zeros(self.getNumGames())
    for n in range(self.getNumGames()):
      print("going through..." + str(n))
      currState = self.getEnv().reset()
      utility = 0 
      currReward = 0
      done = False
      while not done:
        prevAction = self.greedyPolicy(allStates(ImmutableMatrix(currState)), epsilon)
        nextState, reward, done, info = self.getEnv().step(prevAction)
        if reward is not None:
          prevReward = reward
          utility = utility + reward

        prevState = currState
        currState = nextState

        nextAction = self.greedyPolicy(ImmutableMatrix(currState), epsilon)

        self.updateQ(allStates(ImmutableMatrix(prevState)), prevAction, reward, allStates(ImmutableMatrix(currState)), nextAction, done, alpha, gamma)
        

      epsilon = self.getNewEpsilon(epsilon, finalEpsilon, epsilonDrop) 
      rewards[n] = utility      

    return rewards

def plotRewards(rewards):
  n = len(rewards)
  running_avg = np.empty(n)
  for t in range(n):
    running_avg[t] = np.mean(rewards[max(0, t-100):(t+1)])
  plt.plot(running_avg)
  plt.title("Reward Averages")
  plt.xlabel('iteration', fontsize=18)
  plt.ylabel('utility', fontsize=16)
  plt.show()

numGames = int(sys.argv[1])
s = Sarsa(numGames)
r = s.run()
plotRewards(r)

