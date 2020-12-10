"""Geodesy-related utility functions."""
import importlib
import os

import numpy as np
import pyproj

from RAiDER import Geo2rdr
from RAiDER.mathFcns import cosd, sind
from RAiDER.constants import Zenith


def lla2ecef(lat, lon, height):
    ecef = pyproj.Proj(proj='geocent')
    lla = pyproj.Proj(proj='latlong')

    return pyproj.transform(lla, ecef, lon, lat, height, always_xy=True)


def enu2ecef(east, north, up, lat0, lon0, h0):
    """Return ecef from enu coordinates."""
    # I'm looking at
    # https://github.com/scivision/pymap3d/blob/master/pymap3d/__init__.py
    x0, y0, z0 = lla2ecef(lat0, lon0, h0)

    t = cosd(lat0) * up - sind(lat0) * north
    w = sind(lat0) * up + cosd(lat0) * north

    u = cosd(lon0) * t - sind(lon0) * east
    v = sind(lon0) * t + cosd(lon0) * east

    my_ecef = np.stack((x0 + u, y0 + v, z0 + w))

    return my_ecef


def _get_g_ll(lats):
    '''
    Compute the variation in gravity constant with latitude
    '''
    # TODO: verify these constants. In particular why is the reference g different from self._g0?
    return 9.80616 * (1 - 0.002637 * cosd(2 * lats) + 0.0000059 * (cosd(2 * lats))**2)


def _get_Re(lats):
    '''
    Returns the ellipsoid as a fcn of latitude
    '''
    # TODO: verify constants, add to base class constants?
    Rmax = 6378137
    Rmin = 6356752
    return np.sqrt(1 / (((cosd(lats)**2) / Rmax**2) + ((sind(lats)**2) / Rmin**2)))


def _geo_to_ht(lats, hts, g0=9.80556):
    """Convert geopotential height to altitude."""
    # Convert geopotential to geometric height. This comes straight from
    # TRAIN
    # Map of g with latitude (I'm skeptical of this equation - Ray)
    g_ll = _get_g_ll(lats)
    Re = _get_Re(lats)

    # Calculate Geometric Height, h
    h = (hts * Re) / (g_ll / g0 * Re - hts)

    return h

def checkShapes(los, lats, lons, hts):
    '''
    Make sure that by the time the code reaches here, we have a
    consistent set of line-of-sight and position data.
    '''
    if los is None:
        los = Zenith
    test1 = hts.shape == lats.shape == lons.shape
    try:
        test2 = los.shape[:-1] == hts.shape
    except AttributeError:
        test2 = los is Zenith

    if not test1 and test2:
        raise ValueError(
            'I need lats, lons, heights, and los to all be the same shape. ' +
            'lats had shape {}, lons had shape {}, '.format(lats.shape, lons.shape) +
            'heights had shape {}, and los was not Zenith'.format(hts.shape))


def checkLOS(los, Npts):
    '''
    Check that los is either:
       (1) Zenith,
       (2) a set of scalar values of the same size as the number
           of points, which represent the projection value), or
       (3) a set of vectors, same number as the number of points.
     '''
    # los is a bunch of vectors or Zenith
    if los is not Zenith:
        los = los.reshape(-1, 3)

    if los is not Zenith and los.shape[0] != Npts:
        raise RuntimeError('Found {} line-of-sight values and only {} points'
                           .format(los.shape[0], Npts))
    return los


