import numpy as np
import collections
import functools
import LossFuncs as LF
from Layers import Dense, Conv, Stack, Flatten, Pool

def CostFunc(W0, layers, A, y, Loss):
    num_layers = len( layers )
    
    # Eventually merge with layer below
    m = 0
    for i in range( num_layers ):
        if isinstance(layers[i], (Dense, Conv, Stack)): 
            n = m + layers[i].getNumWeights()
            layers[i].setWeights( W0[m:n] )
            m = n

    for i in range( num_layers ):
        A_next = layers[i].prop( A )
        A = A_next
    
    J, dJdA = Loss( A, y )

    GradList = []
    for i in range( num_layers-1, -1, -1 ):
        if isinstance( layers[i], (Dense, Conv) ):
            dJdA_prev, dJdW, dJdb = layers[i].backprop(dJdA)
            dJdA = dJdA_prev
            GradList.append( dJdb.flatten() )
            GradList.append( dJdW.flatten() )
        elif isinstance( layers[i], Stack ):
            dJdA_prev, Grad = layers[i].backprop(dJdA)
            dJdA = dJdA_prev
            GradList.append( Grad )
        elif isinstance( layers[i], (Flatten, Pool)):
            dJdA_prev = layers[i].backprop(dJdA)
            dJdA = dJdA_prev

    Grad = np.concatenate( GradList[::-1] )

    return J, Grad

def FuncWrapper(f, cache_size=10):
    evals = {}
    last_points = collections.deque()

    def get(pt, which):
        s = pt.tostring()
        if s not in evals:
            evals[s] = f(pt)
            last_points.append(s)
            if len(last_points) >= cache_size:
                del evals[last_points.popleft()]
        return evals[s][which]
    return functools.partial(get, which=0), functools.partial(get, which=1)

class CustomError(Exception):
    pass