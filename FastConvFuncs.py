import numpy as np

def Conv_slice( As, W, b):
    Z = np.dot(As.flatten(), W.flatten()) + b
    #Z = np.sum(np.multiply(As, W), axis=None) + b
    return Z

def Conv(A_prev, W, b, h_params):
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape

    stride = h_params['stride']
    pad = h_params['pad']

    pad = checkPad(pad, stride, f, n_W_prev)

    n_H = int( ( n_H_prev - f + 2*pad ) / stride ) + 1
    n_W = int( ( n_W_prev - f + 2*pad ) / stride ) + 1

    A_prev_pad = zeroPad(A_prev, pad)
    Z = np.zeros((m, n_H, n_W, n_C))

    
    A_col = im2col(A_prev_pad, f, stride)
    W_col = W.reshape(n_C, f*f*n_C_prev)
    for i in range(m):
        Z_col = W_col @ A_col[i, :, :].transpose()
        Z[i, :, :, :] = Z_col.reshape(1, n_H, n_W, n_C) + b

    cache = (A_prev, W, b, h_params)

    assert(Z.shape == (m, n_H, n_W, n_C))

    return Z, cache

def dConv(dJdZ, cache):
    A_prev, W, b, h_params = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = h_params['stride']
    pad = h_params['pad']

    pad = checkPad(pad, stride, f, n_W_prev)

    m, n_H, n_W, n_C = dJdZ.shape

    dW = np.zeros([f, f, n_C_prev, n_C])

    A_prev_pad = zeroPad(A_prev, pad)

    db = np.sum(dJdZ, axis=(0, 1, 2))

    A_col = im2col(A_prev_pad, f, stride)
    W_col = W.reshape(n_C, f*f*n_C_prev)

    dJdZ = dJdZ.reshape(m, n_C, -1)
    for i in range(m):
        dW_col = dJdZ[i, :, :] @ A_col[i, :, :]
        dW += dW_col.reshape(dW.shape)

    dJdA_col = W_col.transpose() @ dJdZ
    dA_prev = col2im(dJdA_col, A_prev_pad.shape, f, pad, stride)

    return dA_prev, dW, db

def indices(A_shape, f, s=1):
    m, n_H_prev, n_W_prev, n_C_prev = A_shape

    n_H = int((n_H_prev - f) / s + 1)
    n_W = int((n_W_prev - f) / s + 1)

    i0 = np.tile(np.repeat(np.arange(f), f), n_C_prev)
    i1 = s * np.repeat(np.arange(n_H), n_W)
    
    j0 = np.tile(np.arange(f), f * n_C_prev)
    j1 = s * np.tile(np.arange(n_W), n_H)
    
    i = i0.reshape(1, -1) + i1.reshape(-1, 1)
    j = j0.reshape(1, -1) + j1.reshape(-1, 1)

    k = np.repeat(np.arange(n_C_prev), f**2).reshape(1, -1)

    return (i, j, k)

def im2col(A, f, s):
    i, j, k = indices(A.shape, f, s)
    A_col = A[:, i, j, k]
    
    return A_col

def col2im(dJdA_col, dJdA_shape, f, pad, s):
    m, n_H, n_W, n_C = dJdA_shape
    
    i, j, k = indices(dJdA_shape, f, s)
    dJdA = np.zeros( (dJdA_shape) )

    dJdA_col = dJdA_col.reshape(m, n_C * f ** 2, -1 )
    dJdA_col = dJdA_col.transpose((0, 2, 1))

    np.add.at(dJdA, (slice(None), i, j, k), dJdA_col)
    if pad == 0:
        return dJdA
    else:
        return dJdA[:, pad:-pad, pad:-pad, :]

def zeroPad(A, pad):

    A_pad = np.pad(A, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))

    return A_pad

def checkPad(pad, s, f, n):
    if not isinstance(s, int):
        raise CustomError('Invalid stride value.')
    
    elif pad == 'same':
        if np.mod(f + n*s - n - s, 2) == 1:
            raise CustomError('For same padding, (f + n*s - n - s) must be even.')
        else:
            pad = int( (f + n*s - n - s) / 2 )
    
    elif not isinstance(pad, int):
        raise CustomError('Invalid pad value.')
    
    return pad


class CustomError(Exception):
    pass