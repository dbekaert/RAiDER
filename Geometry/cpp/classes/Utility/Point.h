/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#ifndef POINT_H
#define POINT_H

#include <iostream>
#include <math.h>
#include <vector>

class pixel {
    /*A structure to define a pixel in radar range-Doppler coordinate system
 *      * represented by time (seconds) and range (slant range in meters);
 *           */
    public:

        double time;
        double range;
    public:

        pixel () { time=0; range=0;}
        pixel (double a, double b) { time=a; range=b; }
        pixel (double *a) { time=a[0]; range=a[1]; }
        ~pixel () {}

};

class point {
  /* A class representing a 3D point with x, y and z coordinates.
 *    */
  friend std::ostream& operator<< (std::ostream&, point);
  friend std::istream& operator>> (std::istream&, point &);

  public: // members

    double x;
    double y;
    double z;

  public :

    point () { x=0; y=0; z=0; }
    point (double a, double b, double c) { x=a; y=b; z=c; }
    point (double *a) { x=a[0]; y=a[1]; z=a[2]; }
    ~point () {}

    point unity ();
    double magnitude () { return sqrt(x*x + y*y + z*z); }
    

};

#endif


