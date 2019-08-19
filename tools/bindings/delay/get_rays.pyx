import numpy as np

DTYPE = np.float64


def compute_ray(double length, double[:] start_pos, double[:] scaled_look_vec, int stepSize):
    '''
    Cython version of _compute_ray
    '''

    # initialize the ray
    ray_path = np.arange(0, length, stepSize)
    cdef Py_ssize_t N = len(ray_path)
    ray = np.zeros((N,3), dtype=DTYPE)

    # type declarations
    cdef:
       Py_ssize_t j
       double sp_0  = start_pos[0]
       double sp_1  = start_pos[1]
       double sp_2  = start_pos[2]
       double slv_0 = scaled_look_vec[0]
       double slv_1 = scaled_look_vec[1]
       double slv_2 = scaled_look_vec[2]
       double[:,:] ray_view = ray
       double[:] rp_view = ray_path

    # iterate through the ray and compute the points
    for j in range(N): 
       ray_view[j,0] = getPos(sp_0, rp_view[j], slv_0)
       ray_view[j,1] = getPos(sp_1, rp_view[j], slv_1)
       ray_view[j,2] = getPos(sp_2, rp_view[j], slv_2)
       
    return ray

cdef double getPos(double sp, double rp, double slv):
     return sp + rp*slv
