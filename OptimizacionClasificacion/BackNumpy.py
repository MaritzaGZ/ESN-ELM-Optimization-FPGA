import numpy as np
import math

Zeros = np.zeros
Ones = np.ones
Empty = np.empty
Dot = np.dot
Log = np.log
Exp = np.exp
Pinv = np.linalg.pinv
Sqrt = np.sqrt
Mean = np.mean
Vs = np.vstack
Array = np.array
Seed = np.random.seed
TanH = np.tanh
Random = np.random.rand
ArgMax = np.argmax
Max = np.max
Abs = np.abs
Eigen = np.linalg.eig
Permutation = np.random.permutation
Tile = np.tile
ptp = np.ptp

def activation_relu(x):
    return np.where(x > 0, x, 0)

def activation_leaky_relu(x):
    return np.where(x >= 0, x, x/8)

def inv_activation_leaky_relu(x):
    return np.where(x >= 0, x, x*8)