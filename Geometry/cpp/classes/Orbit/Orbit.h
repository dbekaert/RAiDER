/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#ifndef ORBIT_H
#define ORBIT_H

#include <iostream>
#include <math.h>
#include <vector>
#include "StateVector.h"

using namespace std;
typedef std::vector< double > dVec;

class Orbit{

    /* A class to reprtesent orbit of a platform. Orbit object (an instance 
      of this class) contains state vectors in discrete time. 
      The class contains a method for interpolating the orbit and a method 
      to extract state vector at a given time from interpolated orbit.
     */

    public:
        
        Orbit(void) {};
        ~Orbit(void) {};

        dVec t;
        dVec x;
        dVec y;
        dVec z;
        dVec vx;
        dVec vy;
        dVec vz;
        
        dVec dx;
        dVec dy;
        dVec dz;
        dVec dvx;
        dVec dvy;
        dVec dvz;

    public:
        Orbit (dVec&, dVec&, dVec&, dVec&, dVec&, dVec&, dVec&);
        void populate(dVec&, dVec&, dVec&, dVec&, dVec&, dVec&, dVec&);
        void interpolate();
        statevector get_statevector(double&);
    private:
        std::vector<double> spline(std::vector<double>&, std::vector<double>&);
        double splint(std::vector<double>&, std::vector<double>&, std::vector<double>&, double&);

};

#endif

