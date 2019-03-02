import numpy as np
import Nueral
import Layers
import ActivationFuncs as ActF
import LossFuncs as LF

NN = Nueral.Network()

np.random.seed(1)
X = np.random.rand(1, 256, 256, 3)
y = np.random.rand(1, 1)
NN.loadData(X, y)

NN.setLoss = LF.L2

L0 = Layers.Conv( f=3, n_C_prev= 3, n_C=10, pad=0, stride=2 )
L1 = Layers.Pool( f=2, mode='max', stride=2 )
L2 = Layers.Conv( f=3, n_C_prev=10, n_C=20, pad=0, stride=2 )
L3 = Layers.Pool( f=2, mode='max', stride=2  )
L4 = Layers.Flatten()
L5 = Layers.Dense( 4500, 10 )
L6 = Layers.Dense( 10, 1 )

NN.addLayer( L0 )
NN.addLayer( L1 )
NN.addLayer( L2 )
NN.addLayer( L3 )
NN.addLayer( L4 )
NN.addLayer( L5 )
NN.addLayer( L6 )

NN.printLayer()

#NN.checkGrad()

NN.train(maxIter=10)