import numpy as np


def relu(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [max(v,0) for v in a]
    a = np.reshape(a, a_shape)
    return a
    
def sigmoid(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [1 / (1 + np.exp(-v)) for v in a]
    a = np.reshape(a, a_shape)
    return a

def tanh(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [np.tanh(v) for v in a]
    a = np.reshape(a, a_shape)
    return a

def leaky_relu(a):
    a_shape = a.shape
    a = np.reshape(a,[-1])
    a = [max(0.1*v,v) for v in a]
    a = np.reshape(a, a_shape)
    return a
