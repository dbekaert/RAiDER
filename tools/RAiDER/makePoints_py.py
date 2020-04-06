import  numpy as np

DTYPE = np.float64

def makePoints3D_py(M, stepSize, Rays_SP, Rays_SLV):
    '''
    Numpy version of cython code for testing purposes 
    '''
    nrow = Rays_SP.shape[0]
    ncol = Rays_SP.shape[1]
    ray = np.zeros((nrow, ncol), 3, M)
    basespace = np.arange(0, M, stepSize) # M = max_len+stepSize

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k3 in [0,1,2]:
                for k4 in range(M):
                    ray[k1,k2,k3,k4] = Rays_SP[k1,k2,k3] + basespace[k4]*Rays_SLV[k1,k2,k3]
    return ray
