/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#ifndef ELLIPSOID_H
#define ELLIPSOID_H

struct ellipsoid {
    /* A structure representing an ellipsoid which approximates a planet.
 *      */
    double a;
    double b;
    double f;
    double e2 ;
    ellipsoid () {a=0.0;b=0.0;f=0.0; e2 = 0.0;};
    ellipsoid (double x, double y) {a=x;b=y;f = (a-b)/a; e2 = 2.0*f-f*f;};
    void wgs84() {a=6378137.0;b=6356752.314245312;f=(a-b)/a; e2 = 2.0*f-f*f;};
    void info(){
    cout << "ellipsoid:" << endl;
    cout << "semi major axis a: " << a << endl;
    cout << "semi minor axis b: " << b << endl;
    cout << "eccentricity e2: " << e2 << endl;
    }

};

#endif
