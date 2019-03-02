import numpy as np

def L1(h, y):
    m = h.shape[0]
    # J = sum( abs( h - y ) )
    dJ  = h - y
    J  = np.sum( np.abs( dJ.flatten() ) ) / m
        
    dJ[ dJ<0 ] = -1/m
    dJ[ dJ>0 ] =  1/m
    return J, dJ

def L2(h, y):
    m = h.shape[0]
    # J = (1/2) * sum( h - y ).^2
    J = np.sum( pow(( h.flatten() - y.flatten() ), 2) ) / (2*m)

    dJ = (h - y) / m
    return J, dJ

def Logisitic( h, y ):
    pass

def TripleMag( h, y ):
    pass