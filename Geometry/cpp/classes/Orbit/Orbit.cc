/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#include <iostream>
#include <math.h>
#include <vector>
#include "Orbit.h"

using namespace std;
typedef std::vector< double > dVec;



void Orbit::populate(dVec &T, dVec &X, dVec &Y, dVec &Z, dVec &Vx, dVec &Vy, dVec &Vz)
{
    t = T;
    x = X;
    y = Y;
    z = Z;
    vx = Vx;
    vy = Vy;
    vz = Vz;

}

void Orbit::interpolate()
{
    dx = Orbit::spline(t, x);
    dy = Orbit::spline(t, y);
    dz = Orbit::spline(t, z);
    dvx = Orbit::spline(t, vx);
    dvy = Orbit::spline(t, vy);
    dvz = Orbit::spline(t, vz);

}   

statevector Orbit::get_statevector(double &target_time)
{
    statevector st;
    st.position.x = Orbit::splint(t, x, dx, target_time);
    st.position.y = Orbit::splint(t, y, dy, target_time);
    st.position.z = Orbit::splint(t, z, dz, target_time);
    st.velocity.x = Orbit::splint(t, vx, dvx, target_time);
    st.velocity.y = Orbit::splint(t, vy, dvy, target_time);
    st.velocity.z = Orbit::splint(t, vz, dvz, target_time);

    st.t = target_time;
    return st;
}

std::vector<double> Orbit::spline(std::vector<double> &x, std::vector<double> &y)
{
    int n = x.size();
    double sig, p;

    std::vector<double> y2(n, 0);
    std::vector<double> u(n, 0);
    y2[0] = 0;
    u[0] = 0;


    for (int i=1; i<n; i++)
    {
        sig=(x[i]-x[i-1])/(x[i+1]-x[i-1]);
        p=sig*y2[i-1]+2.0;
        y2[i]=(sig-1.)/p;
        u[i]=(6.*((y[i+1]-y[i])/(x[i+1]-x[i])-(y[i]-y[i-1])/(x[i]-x[i-1]))/(x[i+1]-x[i-1])-sig*u[i-1])/p;

    }

    for (int k = n-2; k>0; k--)
    {
        y2[k] = y2[k]*y2[k+1] + u[k];
    }

    return y2;

}

double Orbit::splint(std::vector<double> &xa, std::vector<double> &ya, std::vector<double> &y2a, double &x)
{
    int n = xa.size();
    int klo=0;
    int khi=n-1;
    int k;
    while (khi-klo > 1)
    {
        k = int((khi+klo)/2);

        if (xa[k]>x){
           khi=k;
        }
        else{
           klo=k;
        };
    }


    double h = xa[khi]-xa[klo];
    double a = (xa[khi]-x)/h;
    double b = (x-xa[klo])/h;
    double y = a*ya[klo] + b*ya[khi] + ((pow(a,3)-a)*y2a[klo] + (pow(b,3)-b)*y2a[khi])*(pow(h,2))/6;

    return y;

}

