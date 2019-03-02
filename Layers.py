import numpy as np
import ConvolutionFuncs as ConvF
import ActivationFuncs as ActF
import PoolingFuncs as PoolF

# TODO
# Input/Output Dimension Checking

class Flatten():
    def __init__(self):
        self.shape = ()

    def Type(self):
        return 'Flatten'

    def outputShape(self, inputShape):
        m, n_H, n_W, n_C = inputShape
        outputShape = (m, n_H * n_W * n_C)
        return outputShape

    def getNumWeights(self):
        return 0

    def setWeights(self, W0):
        pass

    def prop(self, A_prev):
        self.shape = A_prev.shape
        m, n_H, n_W, n_C = self.shape
        A = A_prev.reshape(m, n_H * n_W * n_C)

        return A

    def backprop(self, dJdA):
        dJdA_prev = dJdA.reshape(self.shape)

        return dJdA_prev

class Dense():
    def __init__(self, n_prev, n_hidden, activation=ActF.sigmoid):
        self.n_hidden = n_hidden
        self.n_prev   = n_prev
        self.activation = activation
        self.W = []
        self.b = []

    def Type(self):
        return 'Dense'

    def outputShape(self, inputShape):
        m, n_prev = inputShape
        if n_prev == self.n_prev:
            outputShape = (m, self.n_hidden)
        else:
            raise ValueError('Input shape mismatch')
            
        return outputShape

    def getNumWeights(self):
        numWeights = self.n_hidden * (1 + self.n_prev)
        return numWeights
    
    def setWeights(self, W0):
        self.W = W0[0:self.n_hidden*self.n_prev].reshape(self.n_prev, self.n_hidden)
        self.b = W0[self.n_hidden*self.n_prev:].reshape(1, self.n_hidden)

    def prop(self, A_prev):
        self.A_prev = A_prev
        Z = A_prev @ self.W + self.b
        self.A, self.dAdZ = self.activation(Z)
        
        return self.A

    def backprop(self, dJdA):
        dJdZ = dJdA * self.dAdZ
        
        dJdW = self.A_prev.transpose() @ dJdZ
        dJdb = sum(dJdZ)

        dJdA = dJdZ @ self.W.transpose() 

        return dJdA, dJdW, dJdb

class Conv():
    def __init__( self, f=3, n_C=3, n_C_prev=3, pad='same', stride=1, activation=ActF.sigmoid ):

        h_params = {'pad':pad, 'stride':stride}

        self.f = f
        self.n_C = n_C
        self.n_C_prev = n_C_prev
        self.numWeights = f**2 * n_C_prev * n_C + n_C
        self.activation = activation
        self.cache = []
        self.W = []
        self.b = []
        self.h_params = h_params
        self.dAdZ = []
    
    def Type(self):
        return 'Conv'

    def outputShape(self, inputShape):
        m, n_H_prev, n_W_prev, n_C_prev = inputShape
        
        if n_C_prev != self.n_C_prev:
            raise ValueError('Input shape mismatch. (n_C_prev)')

        stride = self.h_params['stride']
        pad = self.h_params['pad']

        pad = ConvF.checkPad(pad, stride, self.f, n_W_prev)
        n_H = int( ( n_H_prev - self.f + 2*pad ) / stride ) + 1
        n_W = int( ( n_W_prev - self.f + 2*pad ) / stride ) + 1
        
        outputShape = (m, n_H, n_W, self.n_C)
        return outputShape

    def getNumWeights(self):
        return self.numWeights

    def setWeights(self, W0):
        self.W = W0[0:self.f**2*self.n_C_prev*self.n_C].reshape(self.f, self.f, self.n_C_prev, self.n_C)
        self.b = W0[self.f**2*self.n_C_prev*self.n_C:].reshape(1, 1, 1, self.n_C)

    def prop(self, A_prev):
        Z, self.cache = ConvF.Conv(A_prev, self.W, self.b, self.h_params)
        A, self.dAdZ = self.activation(Z)

        return A

    def backprop(self, dJdA):
        dJdZ = dJdA * self.dAdZ
        dJdA_prev, dJdW, dJdb = ConvF.dConv(dJdZ, self.cache)

        return dJdA_prev, dJdW, dJdb

class Pool():
    def __init__( self, f = 2, stride = 1, mode='max' ):
        self.mode = mode
        self.h_params = {'f':f, 'stride':stride}

    def outputShape(self, inputShape):
        (m, n_H_prev, n_W_prev, n_C_prev) = inputShape

        f = self.h_params['f']
        stride = self.h_params['stride']

        n_H = int( 1 + (n_H_prev - f) / stride )
        n_W = int( 1 + (n_W_prev - f) / stride )

        outputShape = (m, n_H, n_W, n_C_prev)

        return outputShape

    def Type(self):
        return 'Pool'

    def getNumWeights(self):
        return 0

    def setWeights(self, W0):
        pass

    def prop(self, A_prev):
        A, self.cache = PoolF.Pool(A_prev, self.h_params, self.mode)

        return A

    def backprop(self, dJdA):
        dJdA = PoolF.dPool(dJdA, self.cache)

        return dJdA

class Stack():
    def __init__( self, layers ):
        self.layers = layers
        self.num_layers = len( layers )
        self.A_prev = []
        self.n_C = []

    def Type(self):
        return 'Stack'

    def getNumWeights(self):
        numWeights = 0
        for i in range( self.num_layers ):
            numWeights += self.layers[i].getNumWeights()

        return numWeights

    def setWeights(self, W0):
        m = 0
        for i in range( self.num_layers ):
            if isinstance(self.layers[i], (Dense, Conv)): 
                n = m + self.layers[i].getNumWeights()

                self.layers[i].setWeights( W0[m:n] )
                m = n

    def prop(self, A_prev):
        self.A_prev = A_prev

        A_list = []
        for i in range( self.num_layers ):
            A_layer = self.layers[i].prop(A_prev)
            A_list.append(A_layer)
            self.n_C.append(A_layer.shape[3])
        
        A = np.concatenate(A_list, 3)
        

        return A

    def backprop(self, dJdA):
        GradList = []
        dJdA_prev = np.zeros_like(self.A_prev)
        m = 0
        for i in range( self.num_layers ):
            if isinstance(self.layers[i], Conv):
                n = m + self.n_C[i]
                dJdA_prev_layer, dJdW, dJdb = self.layers[i].backprop(dJdA[:, :, :, m:n])
                dJdA_prev += dJdA_prev_layer
                GradList.append( dJdW.flatten() )
                GradList.append( dJdb.flatten() )
                m = n
                
        Grad = np.concatenate( GradList )

        return dJdA, Grad