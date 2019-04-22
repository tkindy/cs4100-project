import gym
from collections import defaultdict, namedtuple
from sympy.matrices import ImmutableMatrix
import numpy as np
import matplotlib.pyplot as plt
import sys

#An implementation of Sarsa specified for Centipede-ram-v0
class Sarsa:

  def __init__(self, numGames):
    self.env = gym.make('Centipede-ram-v0')
    self.qTable = defaultdict(float)
    self.freqTable = defaultdict(int)
    self.numGames = numGames

    self.gamma = 0.99
    self.epsilon = 0.99
    self.finalEpsilon = 0.01
    self.epsilonDrop = (self.epsilon - self.finalEpsilon) / self.numGames * 2
    self.alpha = 0.6

  def greedyPolicy(self, state):
    randomValue = np.random.rand()
    if (randomValue < self.epsilon):
      return self.env.action_space.sample()

    qValues = { i: self.qTable[state, i] for i in range(18) }
    maxQValue = max(qValues.values())

    actions_with_max_q = [a for a, q in qValues.items() if q == maxQValue]
    return np.random.choice(actions_with_max_q)

  def updateQ(self, prevState, nextState, action, nextAction, reward, done):
    qValue = self.qTable[nextState, nextAction]
    self.freqTable[prevState, action] += 1
    update = self.alpha * (reward + self.gamma * qValue * (1 - done)  - self.qTable[prevState, action])
    self.qTable[prevState, action] += update

  def updateEpsilon(self):
    self.epsilon = max(self.epsilon - self.epsilonDrop, self.finalEpsilon)

  def run(self):
    print("RUN %d Games..." % self.numGames)
    utility = 0
    allStates = namedtuple('allStates', ['state'])
    prevState = None
    prevAction = 0
    rewards = np.zeros(self.numGames)
    for n in range(self.numGames):
      print("going through..." + str(n))
      currState = self.env.reset()
      utility = 0
      done = False
      while not done:
        prevAction = self.greedyPolicy(allStates(ImmutableMatrix(currState)))
        nextState, reward, done, _ = self.env.step(prevAction)
        if reward is not None:
          utility += reward

        prevState = currState
        currState = nextState

        nextAction = self.greedyPolicy(ImmutableMatrix(currState))

        self.updateQ(allStates(ImmutableMatrix(prevState)), prevAction, reward, allStates(ImmutableMatrix(currState)), nextAction, done)


      self.updateEpsilon()
      rewards[n] = utility

    return rewards

def plotRewards(rewards):
  n = len(rewards)
  running_avg = [ np.mean(rewards[max(0, t - 100):(t + 1)]) for t in range(n) ]
  plt.plot(running_avg)
  plt.title("Reward Averages")
  plt.xlabel('iteration', fontsize=18)
  plt.ylabel('utility', fontsize=16)
  plt.show()

numGames = int(sys.argv[1])
s = Sarsa(numGames)
r = s.run()
plotRewards(r)
