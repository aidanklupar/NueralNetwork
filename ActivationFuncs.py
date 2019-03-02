import numpy as np

def identity(Z):
    A = Z
    dA = np.ones_like(Z)
    return A, dA

def sigmoid(Z):
    A = 1.0 / (1 + np.exp(-Z)) 
    dA = A * (1 - A)
    return A, dA

def tanh(Z):
    A = np.tanh(Z)
    dA = 1 - np.tanh(Z)**2
    return A, dA

def relu(Z):
    Z_shape = Z.shape
    A = np.max(np.zeros_like(Z.flatten()), Z.flatten()).reshape(Z_shape)
    dA = Z
    dA[ dA<=0 ] = 0
    dA[ dA>0 ] = 1

    return A, dA