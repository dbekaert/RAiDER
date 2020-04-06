import  numpy as np
cimport numpy as cnp

DTYPE = np.float64
ctypedef double       npy_float64
ctypedef npy_float64  float64_t

#cimport cython
#@cython.boundscheck(False)  # turn off array bounds check
#@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints3D(int M, double stepSize, cnp.ndarray[cnp.float64_t, ndim=3] Rays_SP, cnp.ndarray[cnp.float64_t, ndim=3] Rays_SLV):
    '''
    Fast cython code to create the rays needed for ray-tracing
    '''
    #assert Rays_SP.dtype == DTYPE and Rays_SLV.dtype == DTYPE

    cdef int k1, k2, k3, k4
    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=4, mode='c'] ray = np.zeros((nrow, ncol), 3, M)
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode='c'] basespace = np.arange(0, M, stepSize) # max_len+stepSize

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k3 in [0,1,2]:
                for k4 in range(M):
                    ray[k1,k2,k3,k4] = Rays_SP[k1,k2,k3] + basespace[k4]*Rays_SLV[k1,k2,k3]
                    #ray[k1,k2,k3,:] = makeRay(Rays_SP[k1,k2,k3],Rays_SLV[k1,k2,k3], basespace)
    return ray


#cdef makeRay(double sp, double slv_comp, np.ndarray[float64_t, ndim=1] rayPoints):
#    '''
#    Create and return a single ray
#    '''
#    cdef int k
#    cdef int N = len(rayPoints)
#    cdef np.ndarray[float64_t,ndim=1,mode='c'] out = np.zeros((N,))
#    for k in range(N):
#        out[k] = sp + rayPoints[k]*slv_comp
#    return out
