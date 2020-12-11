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


# Part of the following UTM and WGS84 converter is borrowed from https://gist.github.com/twpayne/4409500
# Credits go to Tom Payne
def WGS84_to_UTM(lon, lat, common_center=False):
    shp = lat.shape
    lon = np.ravel(lon)
    lat = np.ravel(lat)
    if common_center == True:
        lon0 = np.median(lon)
        lat0 = np.median(lat)
        z0, l0, x0, y0 = project((lon0,lat0))
    Z = lon.copy()
    L = np.zeros(lon.shape,dtype='<U1')
    X = lon.copy()
    Y = lon.copy()
    for ind in range(lon.__len__()):
        longitude = lon[ind]
        latitude = lat[ind]
        if common_center == True:
            z, l, x, y = project((longitude,latitude), z0, l0)
        else:
            z, l, x, y = project((longitude,latitude))
        Z[ind] = z
        L[ind] = l
        X[ind] = x
        Y[ind] = y
    return (
            np.reshape(Z,shp), 
            np.reshape(L,shp), 
            np.reshape(X,shp), 
            np.reshape(Y,shp)
        )

def UTM_to_WGS84(z, l, x, y):
    shp = x.shape
    z = np.ravel(z)
    l = np.ravel(l)
    x = np.ravel(x)
    y = np.ravel(y)
    lat = x.copy()
    lon = x.copy()
    for ind in range(z.__len__()):
        zz = z[ind]
        ll = l[ind]
        xx = x[ind]
        yy = y[ind]
        coordinates = unproject(zz, ll, xx, yy)
        lat[ind] = coordinates[1]
        lon[ind] = coordinates[0]
    return np.reshape(lon,shp), np.reshape(lat,shp)


def project(coordinates, z=None, l=None):
    if z is None:
        z = zone(coordinates)
    if l is None:
        l = letter(coordinates)
    _proj = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _proj(coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, l, x, y


def unproject(z, l, x, y):
    _proj = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    if l < 'N':
        y -= 10000000
    lng, lat = _proj(x, y, inverse=True)
    return (lng, lat)


def letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


def zone(coordinates):
    if 56 <= coordinates[1] < 64 and 3 <= coordinates[0] < 12:
        return 32
    if 72 <= coordinates[1] < 84 and 0 <= coordinates[0] < 42:
        if coordinates[0] < 9:
            return 31
        elif coordinates[0] < 21:
            return 33
        elif coordinates[0] < 33:
            return 35
        return 37
    return int((coordinates[0] + 180) / 6) + 1


def wgs84():
    ''' Return a dict with WGS-84 info '''
    epsg = 4326
    projname = 'projection'

    # CF 1.8 Convention stuff
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    projds = f.create_dataset(projname, (), dtype='i')
    projds[()] = epsg

    # WGS84 ellipsoid
    attr_dict = {
            'semi_major_axis': 6378137.0,
            'inverse_flattening': 298.257223563,
            'ellipsoid': np.string_("WGS84"),
            'epsg_code': epsg,
            'spatial_ref': np.string_(srs.ExportToWkt()),
            'grid_mapping_name': np.string_('latitude_longitude'),
            'longitude_of_prime_meridian': 0.0,
        }

            x.attrs['standard_name'] = np.string_("longitude")
            x.attrs['units'] = np.string_("degrees_east")
            y.attrs['standard_name'] = np.string_("latitude")
            y.attrs['units'] = np.string_("degrees_north")
            z.attrs['standard_name'] = np.string_("height")
            z.attrs['units'] = np.string_("m")
