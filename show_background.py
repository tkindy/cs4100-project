#!/usr/bin/python3

import numpy as np
import scipy.misc as smp

bg = np.load("background.npy")
img = smp.toimage(bg)
img.show()
