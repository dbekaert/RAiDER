"""Helper utilities for file readers."""


import numpy
import scipy
import util


class LinearModel:
    """Generic weather model.

    This model is based upon a linear interpolation scheme for pressure,
    temperature, and relative humidity.
    """
    def __init__(self, points, pressure, temperature, humidity, k1, k2, k3):
        """Initialize a NetCDFModel."""
        self._p_inp = scipy.interpolate.LinearNDInterpolator(
                points, numpy.log(pressure), fill_value=0)
        self._t_inp = scipy.interpolate.LinearNDInterpolator(
                points, temperature, fill_value=0)
        self._h_inp = scipy.interpolate.LinearNDInterpolator(
                points, humidity, fill_value=0)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def wet_delay(self, a):
        """Calculate delay at a list of points."""
        temperature = self._t_inp(a)
        humidity = self._h_inp(a)
        e = _find_e(temperature, humidity)
        # Let's avoid dividing by 0
        temperature[temperature == 0] = numpy.nan
        wet_delay = self.k2*e/temperature + self.k3*e/temperature**2
        wet_delay[numpy.isnan(wet_delay)] = 0
        return wet_delay

    def hydrostatic_delay(self, a):
        """Calculate hydrostatic delay at a list of points."""
        temperature = self._t_inp(a)
        pressure = numpy.exp(self._p_inp(a))
        temperature[temperature == 0] = numpy.nan
        hydro_delay = self.k1*pressure/temperature
        hydro_delay[numpy.isnan(hydro_delay)] = 0
        return hydro_delay


def _find_e(temp, rh):
    """Calculate partial pressure of water vapor."""
    # From TRAIN:
    # Could not find the wrf used equation as they appear to be
    # mixed with latent heat etc. Istead I used the equations used
    # in ERA-I (see IFS documentation part 2: Data assimilation
    # (CY25R1)). Calculate saturated water vapour pressure (svp) for
    # water (svpw) using Buck 1881 and for ice (swpi) from Alduchow
    # and Eskridge (1996) euation AERKi

    # TODO: figure out the sources of all these magic numbers and move
    # them somewhere more visible.
    svpw = (6.1121
            * numpy.exp((17.502*(temp - 273.16))/(240.97 + temp - 273.16)))
    svpi = (6.1121
            * numpy.exp((22.587*(temp - 273.16))/(273.86 + temp - 273.16)))
    tempbound1 = 273.16 # 0
    tempbound2 = 250.16 # -23

    svp = svpw
    wgt = (temp - tempbound2)/(tempbound1 - tempbound2)
    svp = svpi + (svpw - svpi)*wgt**2
    ix_bound1 = temp > tempbound1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = temp < tempbound2
    svp[ix_bound2] = svpi[ix_bound2]

    e = rh/100 * svp * 100

    return e


def _propagate_down(a):
    """Pull real values down to cover NaNs

    a might contain some NaNs which live under real values. We replace
    those NaNs with actual values. a must be a 3D array.
    """
    out = a.copy()
    z, x, y = out.shape
    for i in range(x):
        for j in range(y):
            held = None
            for k in range(z):
                val = out[k][i][j]
                if numpy.isnan(val) and held is not None:
                    out[k][i][j] = held
                elif not numpy.isnan(val):
                    held = val
    return out


def import_grids(lats, lons, pressure, temperature, temp_fill, humidity,
                 humid_fill, geo_ht, geo_ht_fill, k1, k2, k3):
    """Import weather information to make a weather model object.
    
    This takes in lat, lon, pressure, temperature, humidity in the 3D
    grid format that NetCDF uses, and I imagine might be common
    elsewhere. If other weather models don't make it convenient to use
    this function, we'll need to add some more abstraction. For now,
    this function is only used for NetCDF anyway.
    """
    # In some cases, the pressure goes the wrong way. The way we'd like
    # is from top to bottom, i.e., low pressures to high pressures. If
    # that's not the case, we'll reverse everything.
    if pressure[0] > pressure[1]:
        pressure = pressure[::-1]
        temperature = temperature[::-1]
        humidity = humidity[::-1]
        geo_ht = geo_ht[::-1]
    # Replace the non-useful values by NaN, and fill in values under
    # the topography
    temps_fixed = _propagate_down(numpy.where(temperature != temp_fill,
                                              temperature, numpy.nan))
    humids_fixed = _propagate_down(numpy.where(humidity != humid_fill,
                                               humidity, numpy.nan))
    geo_ht_fix = _propagate_down(numpy.where(geo_ht != geo_ht_fill,
                                             geo_ht, numpy.nan))

    outlength = lats.size * pressure.size
    points = numpy.zeros((outlength, 3), dtype=lats.dtype)
    rows, cols = lats.shape
    def to1D(lvl, row, col):
        return lvl * rows * cols + row * cols + col
    # This one's a bit weird. temps and humids are easier.
    new_plevs = numpy.zeros(outlength, dtype=pressure.dtype)
    for lvl in range(len(pressure)):
        p = pressure[lvl]
        for row in range(rows):
            for col in range(cols):
                new_plevs[to1D(lvl, row, col)] = p
    new_temps = numpy.reshape(temps_fixed, (outlength,))
    new_humids = numpy.reshape(humids_fixed, (outlength,))
    for lvl in range(len(pressure)):
        for row in range(rows):
            for col in range(cols):
                lat = lats[row][col]
                lon = lons[row][col]
                geo_ht = geo_ht_fix[lvl][row][col]
                pt_idx = to1D(lvl, row, col)
                points[pt_idx] = util.toXYZ(lat, lon, geo_ht)

    # So a while ago we removed all the NaNs under the earth, but there
    # are still going to be some left, so we just remove those points.
    points_thing = numpy.zeros(new_plevs.shape, dtype=bool)
    for i in range(new_plevs.size):
        points_thing[i] = numpy.all(numpy.logical_not(numpy.isnan(points[i])))

    ok = util.big_and(numpy.logical_not(numpy.isnan(new_plevs)),
                      numpy.logical_not(numpy.isnan(new_temps)),
                      numpy.logical_not(numpy.isnan(new_humids)), points_thing)

    return LinearModel(points=points[ok], pressure=new_plevs[ok],
                       temperature=new_temps[ok], humidity=new_humids[ok],
                       k1=k1, k2=k2, k3=k3)
