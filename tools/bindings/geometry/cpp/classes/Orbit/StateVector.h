/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#ifndef STATEVECTOR_H
#define STATEVECTOR_H

#include "Point.h"

class statevector {
    /*Class to define a platform statevector, 
     * t: time at which the state vector of the platform is given
     * position: cartesian coordinate of the platform position (x , y, z)
     * velocity: velocity of the platform in all three axes (vx, vy, vz)
     */
    public:
        double t;
        point position;
        point velocity;
};

#endif
