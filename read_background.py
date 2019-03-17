#!/usr/bin/python3

# An example of how to read the background file

import numpy as np

bg = np.load('background.npy')

for y, row in enumerate(bg):
  for x, color in enumerate(row):
    print(color)
