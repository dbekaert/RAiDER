'''

Copyright (c) 2018-
Authors(s): Heresh Fattahi

'''

import cython
import numpy as np
cimport numpy as np
from Geo2rdr cimport Geo2rdr

cimport ArrayWrapper

np.import_array ()

cdef class PyGeo2rdr:
    '''
    Python wrapper for Geo2rdr

    '''
    cdef Geo2rdr* c_geo2rdr

    def __cinit__(self):
         self.c_geo2rdr = new Geo2rdr()

    def __dealloc__ (self):
        del self.c_geo2rdr

   
    def set_geo_coordinate(self, lon_first, lat_first,
                                 lon_step, lat_step,
                                 np.ndarray[double, ndim=2, mode="c"] heights  not None):

        self.c_geo2rdr.set_geo_coordinate ( 
                                          lon_first, lat_first,
                                          lon_step, lat_step,
                                          heights.shape[0], heights.shape[1],
                                          &heights[0,0]) 

    def set_orbit(self, np.ndarray[double,ndim=1,mode="c"] times not None,
                        np.ndarray[double,ndim=1,mode="c"] x not None,
                        np.ndarray[double,ndim=1,mode="c"] y not None,
                        np.ndarray[double,ndim=1,mode="c"] z not None,
                        np.ndarray[double,ndim=1,mode="c"] vx not None,
                        np.ndarray[double,ndim=1,mode="c"] vy not None,
                        np.ndarray[double,ndim=1,mode="c"] vz not None):

        nr_state_vectors = len(x)                
        self.c_geo2rdr.set_orbit(nr_state_vectors,
                                 &times[0],
                                 &x[0],
                                 &y[0],
                                 &z[0],
                                 &vx[0],
                                 &vy[0],
                                 &vz[0])

    def geo2rdr(self):
        self.c_geo2rdr.geo2rdr()

    def get_los(self):
        cdef double* los_x
        cdef double* los_y
        cdef double* los_z
        cdef int[2] dim
        self.c_geo2rdr.get_los(&los_x, &los_y, &los_z, &dim[0], &dim[1])
        np_los_x = ArrayWrapper.pointer_to_double2D (los_x, dim)
        np_los_y = ArrayWrapper.pointer_to_double2D (los_y, dim)
        np_los_z = ArrayWrapper.pointer_to_double2D (los_z, dim)
        ArrayWrapper.numpy_own_array (np_los_x)
        ArrayWrapper.numpy_own_array (np_los_y)
        ArrayWrapper.numpy_own_array (np_los_z)
        return np_los_x, np_los_y, np_los_z

    def get_slant_range(self):
        cdef double* rng
        cdef int[2] dim
        self.c_geo2rdr.get_range(&rng, &dim[0], &dim[1])
        np_rng = ArrayWrapper.pointer_to_double2D (rng, dim)
        ArrayWrapper.numpy_own_array (np_rng)
        return np_rng


