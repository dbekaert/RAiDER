import  numpy as np
cimport numpy as cnp

DTYPE = np.float64
ctypedef double       npy_float64
ctypedef npy_float64  float64_t

cimport cython
@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints3D(double max_len, cnp.ndarray[double, ndim=3] Rays_SP, cnp.ndarray[double, ndim=3] Rays_SLV, double stepSize):
    '''
    Fast cython code to create the rays needed for ray-tracing
    '''
    #assert Rays_SP.dtype == DTYPE and Rays_SLV.dtype == DTYPE

    cdef int k1, k2, k3, k4
    cdef int Npts  = int((max_len+stepSize)//stepSize)
    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef cnp.ndarray[npy_float64, ndim=4, mode='c'] ray = np.empty((nrow, ncol, 3, Npts), dtype=np.float64)
    cdef cnp.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, max_len+stepSize, stepSize) # max_len+stepSize

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k3 in range(3):
                for k4 in range(Npts):
                    ray[k1,k2,k3,k4] = Rays_SP[k1,k2,k3] + basespace[k4]*Rays_SLV[k1,k2,k3]
                    #ray[k1,k2,k3,:] = makeRay(Rays_SP[k1,k2,k3],Rays_SLV[k1,k2,k3], basespace)
    return ray

