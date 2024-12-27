import numpy as np

def ReLU(x) -> float:
    '''Converts real numbers to the range 0 to inf linearly, returning the input if it is positive.'''
    return x if x > 0 else 0

def LeakyReLU(x) -> float:
    '''Converts real numbers to -inf to inf, but negative numbers are divided by 100. Prevents some edge cases with ReLU.'''
    return max(x, 0.01*x)

def Sigmoid(x) -> float:
    '''Converts real values to the range -1 to 1, with values greater than |3| around 0.95'''
    return 1 / (1 + np.exp(-x))

def InputFunc(x) -> float:
    '''Only used for input neurons. Has no effect on the expression, but makes the code cleaner.'''
    return x

activationSelection = [ReLU, LeakyReLU, Sigmoid] # add functions to this as more added.

def randomFunction(rng: np.random.Generator=np.random.Generator()) -> function:
    '''Returns a function to activate neurons. Pass an rng to control seed.'''
    return rng.choice(activationSelection)