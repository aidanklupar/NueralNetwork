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

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zeroPad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, : :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v0 = stride * h
                    vf = v0 + f
                    h0 = stride * w
                    hf = h0 + f

                    a_slice = a_prev_pad[v0:vf, h0:hf]
                    Z[i, h, w, c] = Conv_slice( a_slice, W[:, :, :, c], b[:, :, :, c] )
    
    cache = (A_prev, W, b, h_params)

    assert(Z.shape == (m, n_H, n_W, n_C))

    return Z, cache

def dConv(dZ, cache):
    A_prev, W, b, h_params = cache
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C_prev, n_C = W.shape
    stride = h_params['stride']
    pad = h_params['pad']

    pad = checkPad(pad, stride, f, n_W_prev)

    m, n_H, n_W, n_C = dZ.shape

    dA_prev = np.zeros([m, n_H_prev, n_W_prev, n_C_prev])
    dW = np.zeros([f, f, n_C_prev, n_C])
    db = np.zeros([1, 1, 1, n_C])

    A_prev_pad = zeroPad(A_prev, pad)
    dA_prev_pad = zeroPad(dA_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i, :, :, :]
        da_prev_pad = dA_prev_pad[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v0 = stride * h
                    vf = v0 + f
                    h0 = stride * w
                    hf = h0 + f

                    a_slice = a_prev_pad[v0:vf, h0:hf]

                    da_prev_pad[v0:vf, h0:hf] += W[:, :, :, c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        
        if pad != 0:
            dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
        else:
            dA_prev[i, :, :, :] = da_prev_pad

    return dA_prev, dW, db

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