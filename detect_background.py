#!/usr/bin/python3

# Detect the background for a game given a directory
# of its screenshots and write it to disk

import numpy as np
import sys
from os import listdir
from os.path import join
from collections import defaultdict
import random

screensDir = sys.argv[1]

counts = defaultdict(lambda: defaultdict(lambda: 0))
width, height = int(sys.argv[2]), int(sys.argv[3])

items = listdir(screensDir)
numItems = len(items)
boundary = max(numItems // 100, 1)

for index, item in enumerate(items):
  if index % boundary == 0:
    print("{}% screens processed".format(index / numItems * 100))

  screenFile = join(screensDir, item)
  screen = np.load(screenFile)

  for j in range(height):
    row = screen[j]

    for i in range(width):
      colors = row[i]
      counts[(i, j)][tuple(colors)] += 1

background = np.ndarray((height, width, 3), dtype=int)

for (i, j), colors in counts.items():
  background[j][i] = max(colors, key = lambda x: colors[x])

background.dump('background.npy')
