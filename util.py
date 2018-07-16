"""Geodesy-related utility functions."""


from osgeo import gdal
import numpy as np
import pyproj


def sind(x):
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def tand(x):
    """Return degree tangent."""
    return np.tan(np.radians(x))


def lla2ecef(lat, lon, height):
    ecef = pyproj.Proj(proj='geocent')
    lla = pyproj.Proj(proj='latlong')

    return pyproj.transform(lla, ecef, lon, lat, height)


def ecef2lla(x, y, z):
    ecef = pyproj.Proj(proj='geocent')
    lla = pyproj.Proj(proj='latlong')
    lon, lat, height = pyproj.transform(ecef, lla, x, y, z)
    return lat, lon, height


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


def lla2lambert(lat, lon, height=None):
    lla = pyproj.Proj(proj='latlong')
    lambert = pyproj.Proj('+proj=lcc +lat_1=30.0 +lat_2=60.0 +lat_0=18.500015 +lon_0=-100.2 +a=6370 +b=6370 +towgs84=0,0,0 +no_defs')
    #lambert = pyproj.Proj(proj='lcc')

    if height is None:
        return lla(lat, lon, errcheck=True)
    return pyproj.transform(lla, lambert, lat, lon, height)


def los_to_lv(incidence, heading, lats, lons, heights):
    # I'm looking at http://earthdef.caltech.edu/boards/4/topics/327
    a_0 = incidence
    a_1 = heading
    # TODO: Garbage---must fix
    r = (15000 - heights) / cosd(incidence)

    east = sind(a_0)*cosd(a_1 + 90)
    north = sind(a_0)*sind(a_1 + 90)
    up = cosd(a_0)

    east, north, up = np.stack((east, north, up))*r

    x, y, z = enu2ecef(east.flatten(), north.flatten(), up.flatten(), lats.flatten(), lons.flatten(), heights.flatten())

    los = (np.stack((x, y, z), axis=-1)
            - np.stack(lla2ecef(lats.flatten(), lons.flatten(), heights.flatten()), axis=-1))
    los = los.reshape(east.shape + (3,))
    
    return los


def geo_to_ht(lats, lons, hts):
    """Convert geopotential height to altitude."""
    # Convert geopotential to geometric height. This comes straight from
    # TRAIN
    g0 = 9.80665
    # Map of g with latitude (I'm skeptical of this equation)
    g = 9.80616*(1 - 0.002637*cosd(2*lats) + 0.0000059*(cosd(2*lats))**2)
    Rmax = 6378137
    Rmin = 6356752
    Re = np.sqrt(1/(((cosd(lats)**2)/Rmax**2) + ((sind(lats)**2)/Rmin**2)))

    # Calculate Geometric Height, h
    h = (hts*Re)/(g/g0*Re - hts)

    return h


def toXYZ(lats, lons, hts):
    """Convert lat, lon, geopotential height to x, y, z in ECEF."""
    # Convert geopotential to geometric height. This comes straight from
    # TRAIN
    g0 = 9.80665
    # Map of g with latitude (I'm skeptical of this equation)
    g = 9.80616*(1 - 0.002637*cosd(2*lats) + 0.0000059*(cosd(2*lats))**2)
    Rmax = 6378137
    Rmin = 6356752
    Re = np.sqrt(1/(((cosd(lats)**2)/Rmax**2) + ((sind(lats)**2)/Rmin**2)))

    # Calculate Geometric Height, h
    h = (hts*Re)/(g/g0*Re - hts)
    return lla2ecef(lats, lons, h)


def big_and(*args):
    result = args[0]
    for a in args[1:]:
        result = np.logical_and(result, a)
    return result


def gdal_open(fname):
    ds = gdal.Open(fname)
    val = ds.ReadAsArray()
    # It'll get closed automatically too, but we can be explicit
    del ds
    return val


def gdal_open(fname):
    ds = gdal.Open(fname)
    val = ds.ReadAsArray()
    del ds
    return val
