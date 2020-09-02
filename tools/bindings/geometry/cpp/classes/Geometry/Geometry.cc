/*

Copyright (c) 2018-
Authors(s): Heresh Fattahi

*/

#include "Geometry.h"

void Geo2rdr::set_orbit(int nr_state_vec,
                               double *t,
                               double *x,
                               double *y,
                               double *z,
                               double *vx,
                               double *vy,
                               double *vz)
{

    this->nr_state_vectors = nr_state_vec;
    std::vector<double> zeros(nr_state_vectors, 0);
    orbit.populate(zeros, zeros, zeros, zeros, zeros, zeros, zeros);
    for (int i=0; i<nr_state_vectors; i++)
      {
        orbit.t[i] =  t[i];
        orbit.x[i] =  x[i];
        orbit.y[i] =  y[i];
        orbit.z[i] =  z[i];
        orbit.vx[i] =  vx[i];
        orbit.vy[i] =  vy[i];
        orbit.vz[i] =  vz[i];
      }

    orbit.interpolate();
}

void Geo2rdr::set_geo_coordinate(double lon_first, double lat_first,
                                 double lon_step, double lat_step,
                                 int length, int width,
                                 double* hgt)
{

    this->lon0 = lon_first;
    this->lat0 = lat_first;
    this->delta_lon = lon_step;
    this->delta_lat = lat_step;
    this->nr_lines = length;
    this->nr_pixels = width;

    heights  = new double*[nr_lines];
    heights[0] = hgt;
    for (int line=1; line<nr_lines; line++){
        heights[line] = heights[line-1] + nr_pixels;
    }

}


point Geo2rdr::xyz2llh(point xyz, ellipsoid elip)
{

    // converting lon lat height to cartesian x, y, z

    double f=(elip.a-elip.b)/elip.a;
    double r = sqrt(pow(xyz.x,2) + pow(xyz.y,2) + pow(xyz.z,2));
    double r1 = sqrt(pow(xyz.x,2) + pow(xyz.y,2));
    double Lat = atan2(xyz.z,r1);
    double Lon = atan2(xyz.y, xyz.x);
    double H = r - elip.a;

    for (int i = 0 ; i< 7; i++){
        double N = elip.a/(sqrt(1.0-f*(2.0-f)*pow(sin(Lat),2)));
        double TanLat = xyz.z/r1/(1.0-(2.0-f)*f*N/(N+H));
        Lat = atan2(TanLat,1);
        H = r1/cos(Lat)-N;
        }

    return point(Lon, Lat, H);

}

point Geo2rdr::llh2xyz(point llh, ellipsoid elip)
{
    //converting lon lat height with respect to an ellipsoid
    //to geocentric cartesian coordinates of x, y, z
    //Input llh is expected as an instance of point object with
    //the following components:
    //longitude: llh.x
    //latitude: llh.y
    //height above ellipsoid: llh.z
    //computing longitude radius of curvature (N)
    double R1 = sqrt(1.0-elip.e2*pow(sin(llh.y),2));
    double N = elip.a/R1;

    double x = (N + llh.z)*cos(llh.y)*cos(llh.x);
    double y = (N + llh.z)*cos(llh.y)*sin(llh.x);
    double z = (N*(1-elip.e2) + llh.z)*sin(llh.y);

    return point(x, y, z);

}

pixel Geo2rdr::get_radar_coordinate( Orbit &orbit, point &xyz, double t0)
{
    double t = t0;
    double dt = 1.0;
    double E1, dE1;
    int ii = 0;
    int num_iteration = 20;
    statevector st;

    while (ii < num_iteration)
    {
        st = orbit.get_statevector(t);
        E1 = st.velocity.x*(xyz.x - st.position.x) + st.velocity.y*(xyz.y - st.position.y) + st.velocity.z*(xyz.z - st.position.z);
        dE1 = -pow(st.velocity.x,2) - pow(st.velocity.y,2) - pow(st.velocity.z,2);
        dt = -E1/dE1;
        t = t+dt;
        ii++;
    }
    double t_azimuth = t;

    double slant_range = sqrt(pow(xyz.x - st.position.x ,2) + pow(xyz.y - st.position.y, 2) + pow(xyz.z - st.position.z, 2));
    return pixel(t_azimuth, slant_range);
}

//void Geo2rdr::get_radar_coordinate( Orbit &orbit, point &xyz, point &sensor_xyz, double &slant_range, double t0)
void Geo2rdr::get_radar_coordinate( point &xyz, point &sensor_xyz, double &slant_range, double t0)
{

    double t = t0;
    double dt = 1.0;
    double E1, dE1;
    int ii = 0;
    int num_iteration = 20;
    statevector st;
    double residual_threshold = 0.000000001;
    while (ii < num_iteration && fabs(dt) > residual_threshold)
    {
        //std::cout << ii << std::endl;
        st = orbit.get_statevector(t);
        E1 = st.velocity.x*(xyz.x - st.position.x) + st.velocity.y*(xyz.y - st.position.y) + st.velocity.z*(xyz.z - st.position.z);
        dE1 = -pow(st.velocity.x,2) - pow(st.velocity.y,2) - pow(st.velocity.z,2);
        dt = -E1/dE1;
        t = t+dt;

        //slant_range = sqrt(pow(xyz.x - st.position.x ,2) + pow(xyz.y - st.position.y, 2) + pow(xyz.z - st.position.z, 2));

        //std::cout << setprecision(10) << " ii, dt: " << ii << " , " << dt << " , " << slant_range << std::endl;
        ii++;

    }
    //std::cout << "azimuth time : "  << t << std::endl;
    sensor_xyz.x = st.position.x;
    sensor_xyz.y = st.position.y;
    sensor_xyz.z = st.position.z;
    //double t_azimuth = t;
    slant_range = sqrt(pow(xyz.x - st.position.x ,2) + pow(xyz.y - st.position.y, 2) + pow(xyz.z - st.position.z, 2));

    //double t_isce = 3118.661048;
    //statevector st_i = orbit.get_statevector(t_isce);
    //double rng_isce = sqrt(pow(xyz.x - st_i.position.x ,2) + pow(xyz.y - st_i.position.y, 2) + pow(xyz.z - st_i.position.z, 2));
    //std::cout << setprecision(10) << "range with isce az time: " << rng_isce << std::endl;
}



void Geo2rdr::geo2rdr()
{
    ellipsoid elp;
    elp.wgs84();
    int middle_stvec = nr_state_vectors/2;
    double t0 = orbit.t[5]; // middle_stvec];

    //
    //statevector st = orbit.get_statevector(t0);
    //std::cout << " state vector at : " << t0 << std::endl;
    //std::cout << std::setprecision(10) << st.position.x << " , " << st.position.y << " , "<< st.position.z << std::endl;

    //

    double lat;
    double lon;
    double height;
    point xyz;
    point sensor_xyz;
    pixel rdr_pixel;
    double this_range;
    range = new double[nr_lines*nr_pixels];
    los_x = new double[nr_lines*nr_pixels];
    los_y = new double[nr_lines*nr_pixels];
    los_z = new double[nr_lines*nr_pixels];
    for (int line=0; line < nr_lines; line++){
        for (int pixel=0; pixel<nr_pixels; pixel++){
            lat = lat0 + line*delta_lat;
            lon = lon0 + pixel*delta_lon;
            height = heights[line][pixel];//[line*nr_pixels + pixel];
            xyz = llh2xyz(point(lon, lat, height), elp);
            //rdr_pixel = get_radar_coordinate(orbit, xyz,  t0);
            //get_radar_coordinate(orbit, xyz, sensor_xyz, this_range, t0);
            get_radar_coordinate(xyz, sensor_xyz, this_range, t0);
            //azimuth_time[line*nr_pixels + pixel] = rdr_pixel.time;
            range[line*nr_pixels + pixel] = this_range;
            los_x[line*nr_pixels + pixel] = (sensor_xyz.x - xyz.x)/this_range;
            los_y[line*nr_pixels + pixel] = (sensor_xyz.y - xyz.y)/this_range;
            los_z[line*nr_pixels + pixel] = (sensor_xyz.z - xyz.z)/this_range;

         }
    }

}

void Geo2rdr::get_los(double** ux, double** uy,double** uz, int* length, int* width)
{

    *length = nr_lines;
    *width = nr_pixels;
    *ux = los_x;
    *uy = los_y;
    *uz = los_z;

}

void Geo2rdr::get_sensor_xyz(double** s_x, double** s_y, double** s_z)
{

}

void Geo2rdr::get_range(double** rng, int* length, int* width)
{

    /*double *Rng0 = new double[nr_lines*nr_pixels];
    for (int line = 0; line < nr_lines; line ++) {
        for (int pixel = 0; pixel < nr_pixels; pixel ++) {
            Pha0[line*nr_pixels + pixel] = phase_data[line][pixel];
        }
    }*/
    *length = nr_lines;
    *width = nr_pixels;
    *rng = range;

}

void Geo2rdr::get_azimuth_time(double** az_time)
{

}
