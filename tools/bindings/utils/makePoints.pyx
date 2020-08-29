import  numpy as np
cimport numpy as np

DTYPE = np.float64

cimport cython
from libc.math cimport round as cround
from libc.math cimport NAN

ctypedef double npy_float64
ctypedef float npy_float32
ctypedef npy_float64  float64_t
ctypedef npy_float32  float32_t


def intersect_altitude(pos, dir, altitude):
    """
    Find where rays cast from points on the earth will intersect the given
    altitude. This uses traditional ray tracing equations for ray intersections
    with an ellipsoid.

    :param pos: Ray starting positions in ECEF.
    :param dir: Normalized ray look vector in ECEF.
    :param altitude: Altitude to intersect with in meters.

    :return: Points which lie on the input rays and are located the specified
        altitude.
    """
    # coefficient for semimajor axis
    a = 1 / (6378137.0 + altitude) ** 2
    # coefficient for semiminor axis
    b = 1 / (6356752.3142 + altitude) ** 2

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    dx, dy, dz = dir[..., 0], dir[..., 1], dir[..., 2]

    A = a * (dx ** 2 + dy ** 2) + b * dz ** 2
    B = 2 * a * (x * dx + y * dy) + 2 * b * z * dz
    C = a * (x ** 2 + y ** 2) + b * z ** 2 - 1

    # Quadratic formula
    t = (-B + np.sqrt(B ** 2 - 4 * A * C)) / (2 * A)

    return pos + t.reshape(-1, 1) * dir


@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints0D(double max_len, np.ndarray[double, ndim=1] Rays_SP, np.ndarray[double, ndim=1] Rays_SLV, double stepSize):
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
    if max_len % stepSize != 0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef np.ndarray[npy_float64, ndim = 2, mode = 'c'] ray = np.empty((3, Npts), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim = 1, mode = 'c'] basespace = np.arange(0, max_len+stepSize, stepSize)

    for k3 in range(3):
        for k4 in range(Npts):
            ray[k3, k4] = Rays_SP[k3] + basespace[k4]*Rays_SLV[k3]
    return ray

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def makePoints1D(
    double altitude,
    np.ndarray[npy_float64, ndim=2] start,
    np.ndarray[npy_float64, ndim=2] direction,
    double step_size
):
    '''
    Cast rays along direction until they reach a certain altitude. Ray starting
    points are assumed to be located near the surface of the Earth.

    :param altitude: Maximum ray height in meters measured from the surface of
        the Earth.
    :param start: Ray starting points in ECEF
    :param direction: Ray unit direction vectors in ECEF
    :param step_size: Distance between points on a ray in meters. Must not be 0!
    '''
    cdef np.ndarray[npy_float64, ndim=2] end = intersect_altitude(start, direction, altitude)
    cdef double[:] distances = np.linalg.norm(start - end, axis=-1)
    cdef double maxlen = np.nanmax(distances)
    cdef np.ndarray[npy_float64, ndim=2] sign = np.sign(direction)

    cdef long max_points = (<long>cround(maxlen / step_size)) + 1
    cdef long num_rays = start.shape[0]
    cdef np.ndarray[npy_float64, ndim=3, mode='c'] ray = np.empty((num_rays, 3, max_points), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, maxlen + step_size, step_size)

    cdef double dim_start, dim_end, dim_dir, curr, fill, dim_sign
    cdef long k1, k3, k4, num_points
    for k1 in range(num_rays):
        num_points = (<long>cround(distances[k1] / step_size)) + 1
        for k3 in range(3):
            dim_start = start[k1, k3]
            dim_end = end[k1, k3]
            dim_dir = direction[k1, k3]
            dim_sign = sign[k1, k3]

            # Handle special case where the ray is exactly aligned with an axis
            if dim_dir == 0.:
                for k4 in range(max_points):
                    ray[k1, k3, k4] = 0.
                continue

            # Compute the ray points
            for k4 in range(num_points):
                ray[k1, k3, k4] = dim_start + basespace[k4] * dim_dir

            # Fill the rest of values with 'no data'. This is needed since
            # numpy arrays must be rectangular.
            for k4 in range(num_points, max_points):
                ray[k1, k3, k4] = NAN
    return ray


@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints2D(double max_len, np.ndarray[double, ndim=3] Rays_SP, np.ndarray[double, ndim=3] Rays_SLV, double stepSize):
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
    if max_len % stepSize != 0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef np.ndarray[npy_float64, ndim = 4, mode = 'c'] ray = np.empty((nrow, ncol, 3, Npts), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim = 1, mode = 'c'] basespace = np.arange(0, max_len+stepSize, stepSize)

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k3 in range(3):
                for k4 in range(Npts):
                    ray[k1, k2, k3, k4] = Rays_SP[k1, k2, k3] + basespace[k4]*Rays_SLV[k1, k2, k3]
    return ray


@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
def makePoints3D(double max_len, np.ndarray[double, ndim=4] Rays_SP, np.ndarray[double, ndim=4] Rays_SLV, double stepSize):
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
    if max_len % stepSize != 0:
        Npts = int(max_len//stepSize) + 1
    else:
        Npts = int(max_len//stepSize)

    cdef int nrow = Rays_SP.shape[0]
    cdef int ncol = Rays_SP.shape[1]
    cdef int nz = Rays_SP.shape[2]
    cdef np.ndarray[npy_float64, ndim = 5, mode = 'c'] ray = np.empty((nrow, ncol, nz, 3, Npts), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim = 1, mode = 'c'] basespace = np.arange(0, max_len+stepSize, stepSize)

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k2a in range(nz):
                for k3 in range(3):
                    for k4 in range(Npts):
                        ray[k1, k2, k2a, k3, k4] = Rays_SP[k1, k2, k2a, k3] + basespace[k4]*Rays_SLV[k1, k2, k2a, k3]
    return ray
