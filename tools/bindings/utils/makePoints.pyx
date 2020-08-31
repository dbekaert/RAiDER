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

    cdef long max_points = (<long>cround(maxlen / step_size)) + 1
    cdef long num_rays = start.shape[0]
    cdef np.ndarray[npy_float64, ndim=3, mode='c'] ray = np.empty((num_rays, 3, max_points), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim=1, mode='c'] basespace = np.arange(0, maxlen + step_size, step_size)

    cdef long k1, k3, k4, num_points
    for k1 in range(num_rays):
        num_points = (<long>cround(distances[k1] / step_size)) + 1
        for k3 in range(3):
            # Compute the ray points
            for k4 in range(num_points):
                ray[k1, k3, k4] = start[k1, k3] + basespace[k4] * direction[k1, k3]

            # Fill the rest of values with 'no data'. This is needed since
            # numpy arrays must be rectangular.
            for k4 in range(num_points, max_points):
                ray[k1, k3, k4] = NAN
    return ray


@cython.boundscheck(False)  # turn off array bounds check
@cython.wraparound(False)   # turn off negative indices ([-1,-1])
@cython.cdivision(True)
def makePoints3D(
    double altitude,
    np.ndarray[npy_float64, ndim=4] start,
    np.ndarray[npy_float64, ndim=4] direction,
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

    Output:
      ray: a Nx x Ny x Nz x 3 x Npts array containing the rays tracing a path from the ground pixels, along the
           line-of-sight vectors, up to the maximum length specified.
    '''
    cdef np.ndarray[npy_float64, ndim=2] end = intersect_altitude(start, direction, altitude)
    cdef double[:] distances = np.linalg.norm(start - end, axis=-1)
    cdef double max_len = np.nanmax(distances)

    cdef long max_points = (<long>cround(max_len / step_size)) + 1
    cdef int k1, k2, k2a, k3, k4

    cdef int Npts
    if max_len % step_size != 0:
        Npts = int(max_len//step_size) + 1
    else:
        Npts = int(max_len//step_size)

    cdef int nrow = start.shape[0]
    cdef int ncol = start.shape[1]
    cdef int nz = start.shape[2]
    cdef np.ndarray[npy_float64, ndim = 5, mode = 'c'] ray = np.empty((nrow, ncol, nz, 3, Npts), dtype=np.float64)
    cdef np.ndarray[npy_float64, ndim = 1, mode = 'c'] basespace = np.arange(0, max_len+step_size, step_size)

    for k1 in range(nrow):
        for k2 in range(ncol):
            for k2a in range(nz):
                for k3 in range(3):
                    for k4 in range(Npts):
                        ray[k1, k2, k2a, k3, k4] = start[k1, k2, k2a, k3] + basespace[k4]*direction[k1, k2, k2a, k3]
    return ray
