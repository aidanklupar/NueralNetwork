import numpy as np

def Pool(A_prev, h_params, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    f = h_params['f']
    stride = h_params['stride']

    n_H = int( 1 + (n_H_prev - f) / stride )
    n_W = int( 1 + (n_W_prev - f) / stride )
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v0 = h*stride
                    vf = v0 + f
                    h0 = w*stride
                    hf = h0 + f

                    a_prev_slice = A_prev[i, v0:vf, h0:hf, c]

                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    if mode == "avg":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, h_params)

    assert(A.shape == (m, n_H, n_W, n_C))

    return A, cache

def dPool(dA, cache, mode='max'):
    (A_prev, h_params) = cache
    stride = h_params['stride']
    f = h_params['f']

    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    m, n_H, n_W, n_C = dA.shape

    dA_prev = np.zeros(A_prev.shape)

    for i in range(m):
        a_prev = A_prev[i, :, :, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    v0 = h*stride
                    vf = v0 + f
                    h0 = w*stride
                    hf = h0 + f

                    if mode == "max":
                        a_prev_slice = a_prev[v0:vf, h0:hf, c]
                        mask = create_mask(a_prev_slice)
                        dA_prev[i, v0:vf, h0:hf, c] += mask * dA[i, h, w, c]
                        
                    elif mode == "average":
                        a = dA[i, h, w, c]
                        shape = (f, f)
                        dA_prev[i, v0:vf, h0:hf, c] += distribute_value(a, shape)
    
    assert(dA_prev.shape == A_prev.shape)

    return dA_prev

def create_mask(x):
    mask = (x == np.max(np.max(x)))

    return mask

def distribute_value(dz, shape):
    (n_H, n_W) = shape

    avg = dz / (n_H * n_W)

    a = avg * np.ones(shape)
    
    return a