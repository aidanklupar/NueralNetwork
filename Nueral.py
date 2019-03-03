import numpy as np
import CostFunc as cf
import pickle
from scipy import optimize
import sys
from Layers import Conv, Dense, Stack, Flatten, Pool
import LossFuncs as LF

class Network:
    def __init__(self):
        self.layers = []
        self.Loss = LF.L2
        self.X = []
        self.y = []
        self.W0 = []
        self.WeightsInitialized = 'False'

    def loadData(self, X, y):
        self.X = X
        self.y = y

    def initWeights(self):
        if self.WeightsInitialized == False:
            num_weights = 0
            num_layers = len( self.layers )
            for i in range( num_layers ):
                if isinstance(self.layers[i], (Dense, Conv, Stack)): 
                    num_weights += self.layers[i].getNumWeights()

            self.W0 = np.random.rand(num_weights) * 0.12 * 2 - 0.12

            m = 0
            for i in range( num_layers ):
                if isinstance(self.layers[i], (Dense, Conv, Stack)): 
                    n = m + self.layers[i].getNumWeights()
                    self.layers[i].setWeights( self.W0[m:n] )
                    m = n
            self.WeightsInitialized = True

    def addLayer(self, layer):
        if isinstance(layer, (Conv, Dense, Stack, Flatten, Pool)):
            self.layers.append(layer)
            self.WeightsInitialized = False

    def printLayer(self):
        print('-----Layer Details-----')
        self.initWeights()

        inputShape = self.X.shape
        for i in range( len( self.layers ) ):
            layer = self.layers[i]
            layerType = layer.Type()
            try:
                outputShape = layer.outputShape(inputShape)
                print('Layer', i, '|', layerType, ' Layer |', outputShape , '| Num Weights:', layer.getNumWeights() )
                inputShape = outputShape
            except ValueError:
                print('[-] Input/Output shape mismatch for layer ', i)
                break
        
        if (outputShape != self.y.shape):
            print('[-] Output shape does not match y shape')

        print('Total Num Weights: ', self.W0.size)

    def setLoss(self, func):
        self.Loss = func

    def checkGrad(self):
        self.initWeights()

        eps_val = 1e-6
        dW = np.zeros_like(self.W0)
    
        J = lambda W: cf.CostFunc(self.W0 + W, self.layers, self.X, self.y, self.Loss)[0]
        Grad = lambda W: cf.CostFunc(self.W0 + W, self.layers, self.X, self.y, self.Loss)[1]

        for i in range(dW.size):
            eps = np.zeros_like(self.W0)
            eps[i] = eps_val 

            dW[i] = ( J(+eps) - J(-eps) ) / (2*eps_val)

            if dW[i] == 0:
                pass

        Grad_act = Grad(0)
        
        print('-----Checking Gradient-----')
        all_close = np.allclose(Grad_act, dW)
        print( 'All Close: ',  all_close)
        print( 'Max Error: ', max( abs(Grad_act - dW).flatten() ) )
        print()

    def train(self, maxIter=10, display=False, plot=False, saveRate=50):
        print('-----Training-----')
        self.initWeights()

        # Split Cost and Gradient Functions
        func = lambda x: cf.CostFunc( x, self.layers, self.X, self.y, self.Loss )
        f, fprime = cf.FuncWrapper(func)

        # Define Callback Function
        callback = lambda x: print('Cost: %.4e' % f(x))

        # Optimize
        W_opt, fopt, f_calls, g_calls, flag = optimize.fmin_cg(f=f, fprime=fprime, x0=self.W0, maxiter=maxIter, callback=callback, full_output=True, disp=False)

        self.W0 = W_opt

        if flag == 0:
            print('[+] Minimum Found.')
        elif flag == 1:
            print('[=] Maximum Iterations Reached.')
        elif flag == 2:
            print('[-] Did not converge. Gradient/Cost not changing.')
        
        return