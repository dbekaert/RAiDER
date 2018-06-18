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
        # Log of pressure since pressure is interpolated exponentially
        self._p_inp = scipy.interpolate.LinearNDInterpolator(points,
                                                             numpy.log(
                                                                 pressure))
        self._t_inp = scipy.interpolate.LinearNDInterpolator(points,
                                                             temperature)
        self._rh_inp = scipy.interpolate.LinearNDInterpolator(points, humidity)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

    def pressure(self, x, y, z):
        """Calculate pressure at a point."""
        return numpy.exp(self._p_inp(x, y, z))

    def temperature(self, x, y, z):
        """Calculate temperature at a point."""
        return self._t_inp(x, y, z)

    def rel_humid(self, x, y, z):
        """Calculate relative humidity at a point."""
        return self._rh_inp(x, y, z)


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
            for k in range(z - 1, -1, -1):
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
