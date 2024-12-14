import numpy as np

def ReLU(x) -> float:
    return x if x > 0 else 0

def LeakyReLU(x) -> float:
    return max(x, 0.01*x)

def Sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

activationSelection = [ReLU, LeakyReLU, Sigmoid] # add functions to this as more added.

def randomFunction():
    return np.random.choice(activationSelection)