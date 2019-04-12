#!/usr/bin/python3

import numpy as np
import scipy.misc as smp
from sys import argv

path = argv[1]

bg = np.load(path)
img = smp.toimage(bg)
img.show()
