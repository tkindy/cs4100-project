#!/usr/bin/python3

# Keep a random sample of the screenshots in the given directory,
# removing the rest

from sys import argv
from os import listdir, remove
from os.path import join
import random

screensDir = argv[1]
files = listdir(screensDir)
toKeep = set(random.choices(files, k = 30000))

for f in files:
  if f in toKeep:
    continue

  remove(join(screensDir, f))
