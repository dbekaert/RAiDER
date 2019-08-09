/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <iostream>
#include <iomanip>
#include <assert.h>
#include <math.h>
#include "Orbit.h"
#include "Ellipsoid.h"

using namespace std;

const double PI = 3.141592653589793;

class Geo2rdr
{

    public:
        Geo2rdr(void) {};
        ~Geo2rdr(void) {};

        static Geo2rdr* instance(void)
            {
              if (self == NULL) {self = new Geo2rdr();}
              return self;
            };

        virtual void set_orbit(int nr_state_vectors,
                               double* t,
                               double* x,
                               double* y, 
                               double* z,
                               double* vx,
                               double* vy,
                               double* vz);

        virtual void set_geo_coordinate(double lon_first, double lat_first,
                                        double lon_step, double lat_step,
                                        int length, int width,
                                        double* hgt);

        point xyz2llh(point, ellipsoid);
        point llh2xyz(point, ellipsoid);
        pixel get_radar_coordinate( Orbit &orbit, point &xyz, double t0);
        //virtual void get_radar_coordinate( Orbit& orbit, point& xyz, point& sensor_xyz, double& range, double t0);
        virtual void get_radar_coordinate( point& xyz, point& sensor_xyz, double& range, double t0);

        //void geo2rdr( Orbit &orbit, int nr_state_vectors, double lat0, double lon0, double delta_lat, double delta_lon, double* heights, int nr_lines, int nr_pixels, double* range, double* azimuth);
         
        virtual void geo2rdr();
        virtual void get_los(double** ux, double** uy,double** uz, int* length, int* width);
        virtual void get_sensor_xyz(double** sensor_x, double** sensor_y, double** sensor_z);
        virtual void get_range(double** range, int* length, int* width);
        virtual void get_azimuth_time(double** az_time);
    // ------------------------------------------------
    // data members
        Orbit orbit;
        int nr_state_vectors;
        int nr_lines;
        int nr_pixels;
        double lat0, lon0, delta_lat, delta_lon;
        double **heights;
        double *range; 
        double *azimuth;
        double *los_x;
        double *los_y;
        double *los_z;
        
    private:
        static Geo2rdr* self;

        Geo2rdr(const Geo2rdr&) {assert(false);}

        void operator=(const Geo2rdr&) {assert(false);}

};
#endif
