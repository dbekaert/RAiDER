'''

Copyright (c) 2018-
Authors(s): Heresh Fattahi
  
'''

cdef extern from 'Geometry.h':
    cdef cppclass Geo2rdr:
        Geo2rdr() except +

        void set_orbit(int nr_state_vectors,
                               double* t,
                               double* x,
                               double* y,
                               double* z,
                               double* vx,
                               double* vy,
                               double* vz)
 
        void set_geo_coordinate(double lon_first, double lat_first,
                                 double lon_step, double lat_step,
                                 int length, int width,
                                 double* heights)
       
        void geo2rdr()

        void get_los(double** ux, double** uy,double** uz, int* dim1, int* dim2)

        void get_range(double** rng, int* dim1, int* dim2)


