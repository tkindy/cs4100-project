import gym
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import sys

class Sarsa(object):

  def __init__(self, env, numGames):
    self.env = env
    self.qTable = defaultdict(float)
    self.numGames = numGames

    self.gamma = 0.99
    self.epsilon = 0.99
    self.finalEpsilon = 0.01
    self.epsilonDrop = (self.epsilon - self.finalEpsilon) / self.numGames * 2
    self.alpha = 0.6

  def getNextAction(self, features):
    randomValue = np.random.rand()
    if (randomValue < self.epsilon):
      return self.env.action_space.sample()

    qValues = { i: self.qTable[(features, i)] for i in range(self.env.action_space.n) }
    maxQValue = max(qValues.values())

    actions_with_max_q = [a for a, q in qValues.items() if q == maxQValue]
    return np.random.choice(actions_with_max_q)

  def updateQ(self, prevFeatures, nextFeatures, action, nextAction, reward, done):
    qValue = self.qTable[(nextFeatures, nextAction)]
    prevQValue = self.qTable[(prevFeatures, action)]
    update = self.alpha * (reward + self.gamma * qValue * (1 - done) - prevQValue)
    self.qTable[(prevFeatures, action)] += update

  def updateEpsilon(self):
    self.epsilon = max(self.epsilon - self.epsilonDrop, self.finalEpsilon)

  def run(self):
    print("RUN %d Games..." % self.numGames)
    utility = 0
    prevState = None
    prevAction = 0
    rewards = np.zeros(self.numGames)
    for n in range(self.numGames):
      print("going through..." + str(n))
      currState = self.getFeatureVector(self.env.reset())
      utility = 0
      done = False

      while not done:
        prevAction = self.getNextAction(currState)
        nextState, reward, done, _ = self.env.step(prevAction)
        if reward is not None:
          utility += reward

        prevState = currState
        currState = self.getFeatureVector(nextState)

        nextAction = self.getNextAction(currState)

        self.updateQ(prevState, currState, prevAction, nextAction, reward, done)

      self.updateEpsilon()
      rewards[n] = utility

    return rewards

  def getFeatureVector(self, state):
    raise NotImplementedError("getFeatureVector undefined!")

class SarsaRam(Sarsa):
  def __init__(self, numGames):
    Sarsa.__init__(self, gym.make('Centipede-ram-v0'), numGames)

  def getFeatureVector(self, state):
    return tuple(state)

class SarsaBass(Sarsa):
  def __init__(self, backgroundFile, numGames):
    Sarsa.__init__(self, gym.make('Centipede-v0'), numGames)

    self.background = np.load(backgroundFile, allow_pickle = True)
    self.height = 250
    self.width = 160
    self.m = 10
    self.n = 8
    self.blockHeight = int(self.height / self.m)
    self.blockWidth = int(self.width / self.n)

  def getFeatureVector(self, state):
    print("get")
    convertedState = [ self._convertRow(row, y) for y, row in enumerate(state) ]
    blocks = self._createBlocks(convertedState)
    subvectors = self._createSubvectors(blocks)
    return tuple(subvectors)

  def _convertRow(self, row, y):
    return [ self._convertColor(tuple(color), x, y) for x, color in enumerate(row) ]

  def _convertColor(self, color, x, y):
    if color == tuple(self.background[y][x]):
      return -1

    r, g, b = color
    hashCode = r
    hashCode += (31 * hashCode) + g
    hashCode += (31 * hashCode) + b

    return hashCode % 8

  def _createBlocks(self, state):
    return [ self._createBlock(state, x, y) for x in range(self.m) for y in range(self.n) ]

  def _createBlock(self, state, x, y):
    startX, endX = self._getBounds(x, self.blockWidth)
    startY, endY = self._getBounds(y, self.blockHeight)

    return [ row[startX:endX] for row in state[startY:endY] ]

  def _getBounds(self, index, scale):
    return (index * scale, (index + 1) * scale)

  def _createSubvectors(self, blocks):
    return [ self._createSubvector(block) for block in blocks ]

  def _createSubvector(self, block):
    vector = 0

    for row in block:
      for color in row:
        if color != -1:
          vector |= 1 << color

    return vector

def plotRewards(rewards):
  n = len(rewards)
  running_avg = [ np.mean(rewards[max(0, t - 100):(t + 1)]) for t in range(n) ]
  plt.plot(running_avg)
  plt.title("Reward Averages")
  plt.xlabel('iteration', fontsize=18)
  plt.ylabel('utility', fontsize=16)
  plt.show()

if __name__ == '__main__':
  numGames = int(sys.argv[1])
  s = SarsaBass('../background.npy', numGames)
  r = s.run()
  plotRewards(r)
