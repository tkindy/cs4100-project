import time
import gym
import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class Node:

  def __init__(self, state, reward, discountedReward, childNodes):
    self.state = state
    self.r = reward #The reward received during the simulation node s
    self.v = discountedReward #Cumulated discounted reward of the sub-branch starting from s
    self.c = childNodes #Set of child nodes of s


#def generate_tree():
#1. Iterate through each action
#2. Save each action into the child nodes of the current node
#3. Iterate through those child nodes and repeat the process

#rewards: [(integer, integer),...]
#Plots the rewards
def plotRewards(rewards):
  x = []
  y = []
  for count, reward in rewards:
    x.append(count)
    y.append(reward)
  plt.plot(x, y, 'go-', label='line 1', linewidth=2)
  plt.title("Game Rewards")
  plt.xlabel('iteration', fontsize=18)
  plt.ylabel('utility', fontsize=16)
  plt.show()

#Generates and seraches a tree for an optimal path to play a game
def generateTree(k):
  currentEnv = gym.make('Centipede-ram-v0').env
  #print(type(currentEnv.env))
  currentEnv.reset()
  actions = currentEnv.action_space.n
  state = currentEnv.clone_full_state()
  headNode = Node(state, 0, 0, [])
  queue = deque([headNode])
  count = 0
  max = 0
  rewardList = [(0,0)]
  while len(queue) > 0 and count < k:
    currentNode = queue.popleft()
    for i in range(1, 6):
      currentEnv.restore_full_state(currentNode.state)
      totalRewards = 0
      for j in range(0,14):
        observation, reward, done, info = currentEnv.step(i) 
        totalRewards+= reward
        #TODO: Pick the one with the highest reward...
        if (done):
          break
      if (done):
        continue
      state = currentEnv.clone_full_state()
      #currentEnv.render()
      if (currentNode.r + totalRewards > max):
        max = (currentNode.r + totalRewards)
        rewardList.append((count, max))
        print("Current Max: %d" % (currentNode.r + totalRewards))
      node = Node(state, currentNode.r + totalRewards, 0, [])
      if (count > 100 and node.r < max*0.2):
        #print("skipping...")
        continue
      currentNode.c.append(node)
      queue.append(node)
      #currentEnv.render()
      #time.sleep(0.1)
    count+=1
    if (count % 100 == 0):
      print(count)
  plotRewards(rewardList)

generateTree(50000)

    



  
  
