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

  def getNextAction(self, state):
    features = self.getFeatureVector(state)

    randomValue = np.random.rand()
    if (randomValue < self.epsilon):
      return self.env.action_space.sample()

    qValues = { i: self.qTable[(features, i)] for i in range(self.env.action_space.n) }
    maxQValue = max(qValues.values())

    actions_with_max_q = [a for a, q in qValues.items() if q == maxQValue]
    return np.random.choice(actions_with_max_q)

  def updateQ(self, prevState, nextState, action, nextAction, reward, done):
    prevFeatures = self.getFeatureVector(prevState)
    nextFeatures = self.getFeatureVector(nextState)

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
      currState = self.env.reset()
      utility = 0
      done = False

      while not done:
        prevAction = self.getNextAction(currState)
        nextState, reward, done, _ = self.env.step(prevAction)
        if reward is not None:
          utility += reward

        prevState = currState
        currState = nextState

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

def plotRewards(rewards):
  n = len(rewards)
  running_avg = [ np.mean(rewards[max(0, t - 100):(t + 1)]) for t in range(n) ]
  plt.plot(running_avg)
  plt.title("Reward Averages")
  plt.xlabel('iteration', fontsize=18)
  plt.ylabel('utility', fontsize=16)
  plt.show()

numGames = int(sys.argv[1])
s = SarsaRam(numGames)
r = s.run()
plotRewards(r)
