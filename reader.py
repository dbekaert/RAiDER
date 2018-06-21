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
        e = _find_e(temperature, humidity)
        dry_delay = k2*e/temperature + k3*e/temperature**2
        self._dry_inp = scipy.interpolate.LinearNDInterpolator(
                points, dry_delay, fill_value=0)

        hydro_delay = k1*pressure/temperature
        self._hydro_inp = scipy.interpolate.LinearNDInterpolator(
                points, hydro_delay, fill_value=0)

    def dry_delay(self, a):
        """Calculate delay at a list of points."""
        return self._dry_inp(a)

    def hydrostatic_delay(self, a):
        """Calculate hydrostatic delay at a list of points."""
        return self._hydro_inp(a)


def _find_e(temp, rh):
    """Calculate partial pressure of water vapor."""
    # We have two possible ways to calculate partial pressure of water
    # vapor. There's the equation from Hanssen, and there's the
    # equations from TRAIN. I don't know which is better.

    # Hanssen: (of course, L, latent heat, isn't perfectly accurate.)
    # e_0 = 611.
    # T_0 = 273.16
    # L = 2.5e6
    # R_v = 461.495
    # e_s = e_0*numpy.exp(L/R_v * (1/T_0 - 1/temp))
    # e = e_s * rh / 100

    # From TRAIN:
    # Could not find the wrf used equation as they appear to be
    # mixed with latent heat etc. Istead I used the equations used
    # in ERA-I (see IFS documentation part 2: Data assimilation
    # (CY25R1)). Calculate saturated water vapour pressure (svp) for
    # water (svpw) using Buck 1881 and for ice (swpi) from Alduchow
    # and Eskridge (1996) euation AERKi
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
