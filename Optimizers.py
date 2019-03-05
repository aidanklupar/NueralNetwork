import numpy as np
from scipy import optimize

def MGD(func, W, batch_size):
    alpha = 0.05
    
    for i in range(1, 11):
        J, dJ = func(W) 
        W = W - alpha*dJ
        print('Iter: ', i, ' Cost:', J)

    return J, W

def FMIN_CG(func, W0):
    import CostFunc as cf
    f, fprime = cf.FuncWrapper(func)

    # Define Callback Function
    callback = lambda x: print('Cost: %.4e' % f(x))

    # Optimize
    W_opt, fopt, f_calls, g_calls, flag = optimize.fmin_cg(f=f, fprime=fprime, x0=W0, maxiter=maxIter, callback=callback, full_output=True, disp=False)

    self.W0 = W_opt

    if flag == 0:
        print('[+] Minimum Found.')
    elif flag == 1:
        print('[=] Maximum Iterations Reached.')
    elif flag == 2:
        print('[-] Did not converge. Gradient/Cost not changing.')