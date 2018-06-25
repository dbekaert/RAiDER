'''

Copyright (c) 2018-
Authors(s): Heresh Fattahi

'''
import cython
import numpy as np
cimport numpy as np

np.import_array ()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS (np.ndarray array, int flags)

cdef inline pointer_to_array2D (void* ptr, int* dim, int enum_type):
    cdef np.npy_intp np_dim[2]
    np_dim[0] = <np.npy_intp> dim[0]
    np_dim[1] = <np.npy_intp> dim[1]
    return np.PyArray_SimpleNewFromData (2, np_dim, enum_type, ptr)

cdef inline pointer_to_double2D (void* ptr, int* dim):
    cdef np.ndarray[np.float64_t, ndim=2] array = (
        pointer_to_array2D (ptr, dim, np.NPY_FLOAT64))
    return array

cdef inline numpy_own_array (np.ndarray array):
    PyArray_ENABLEFLAGS (array, np.NPY_OWNDATA)

