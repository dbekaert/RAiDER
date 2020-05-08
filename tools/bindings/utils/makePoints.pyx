import  numpy as np
cimport numpy as cnp

DTYPE = np.float64

cimport cython

ctypedef double npy_float64
ctypedef npy_float64  float64_t

@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints1D(double max_len, cnp.ndarray[double, ndim=1] Rays_SP, cnp.ndarray[double, ndim=1] Rays_SLV, double stepSize):
    '''
    Fast cython code to create the rays needed for ray-tracing. 
    Inputs: 
      max_len: maximum length of the rays
      Rays_SP: 1 x 3 numpy array of the location of the ground pixels in an earth-centered, 
               earth-fixed coordinate system
      Rays_SLV: 1 x 3 numpy array of the look vectors pointing from the ground pixel to the sensor
      stepSize: Distance between points along the ray-path
    Output:
      ray: a 3 x Npts array containing the rays tracing a path from the ground pixels, along the 
           line-of-sight vectors, up to the maximum length specified.
    '''
    cdef int k3, k4
    cdef int Npts  
    if max_len % stepSize !=0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef cnp.ndarray[npy_float64, ndim=2, mode='c'] ray = np.empty((3, Npts), dtype=np.float64)
    cdef cnp.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, max_len+stepSize, stepSize) 

    for k3 in range(3):
        for k4 in range(Npts):
            ray[k3,k4] = Rays_SP[k3] + basespace[k4]*Rays_SLV[k3]
    return ray

@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints2D(double max_len, cnp.ndarray[double, ndim=3] Rays_SP, cnp.ndarray[double, ndim=3] Rays_SLV, double stepSize):
    '''
    Fast cython code to create the rays needed for ray-tracing. 
    Inputs: 
      max_len: maximum length of the rays
      Rays_SP: Nx x Ny x 3 numpy array of the location of the ground pixels in an earth-centered, 
               earth-fixed coordinate system
      Rays_SLV: Nx x Ny x 3 numpy array of the look vectors pointing from the ground pixel to the sensor
      stepSize: Distance between points along the ray-path
    Output:
      ray: a Nx x Ny x 3 x Npts array containing the rays tracing a path from the ground pixels, along the 
           line-of-sight vectors, up to the maximum length specified.
    '''
    cdef int k1, k2, k3, k4

    cdef int Npts  
    if max_len % stepSize !=0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef cnp.ndarray[npy_float64, ndim=4, mode='c'] ray = np.empty((nrow, ncol, 3, Npts), dtype=np.float64)
    cdef cnp.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, max_len+stepSize, stepSize) 

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k3 in range(3):
                for k4 in range(Npts):
                    ray[k1,k2,k3,k4] = Rays_SP[k1,k2,k3] + basespace[k4]*Rays_SLV[k1,k2,k3]
    return ray


cimport cython
@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints3D(double max_len, cnp.ndarray[double, ndim=4] Rays_SP, cnp.ndarray[double, ndim=4] Rays_SLV, double stepSize):
    '''
    Fast cython code to create the rays needed for ray-tracing
    Inputs: 
      max_len: maximum length of the rays
      Rays_SP: Nx x Ny x Nz x 3 numpy array of the location of the ground pixels in an earth-centered, 
               earth-fixed coordinate system
      Rays_SLV: Nx x Ny x Nz x 3 numpy array of the look vectors pointing from the ground pixel to the sensor
      stepSize: Distance between points along the ray-path
    Output:
      ray: a Nx x Ny x Nz x 3 x Npts array containing the rays tracing a path from the ground pixels, along the 
           line-of-sight vectors, up to the maximum length specified.
    '''
    cdef int k1, k2, k2a, k3, k4

    cdef int Npts  
    if max_len % stepSize !=0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef int nz = Rays_SP.shape[2]
    cdef cnp.ndarray[npy_float64, ndim=5, mode='c'] ray = np.empty((nrow, ncol, nz, 3, Npts), dtype=np.float64)
    cdef cnp.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, max_len+stepSize, stepSize) 

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k2a in range(nz):
                for k3 in range(3):
                    for k4 in range(Npts):
                        ray[k1,k2,k2a,k3,k4] = Rays_SP[k1,k2,k2a,k3] + basespace[k4]*Rays_SLV[k1,k2,k2a,k3]
    return ray
