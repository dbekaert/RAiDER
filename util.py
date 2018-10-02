"""Geodesy-related utility functions."""


from osgeo import gdal
import numpy as np
import pickle
import pyproj
import os

gdal.UseExceptions()


# Top of the troposphere
zref = 15000


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
    lambert = pyproj.Proj(
            '+proj=lcc +lat_1=30.0 +lat_2=60.0 +lat_0=18.500015 +lon_0=-100.2 '
            '+a=6370 +b=6370 +towgs84=0,0,0 +no_defs')

    if height is None:
        return lla(lat, lon, errcheck=True)
    return pyproj.transform(lla, lambert, lat, lon, height)


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
    import Geo2rdr

    real_shape = lats.shape
    lats = lats.flatten()
    lons = lons.flatten()
    heights = heights.flatten()

    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)

    loss = np.zeros((3, len(lats)))
    slant_ranges = np.zeros_like(lats)

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        height_array = np.array(((height,),))

        # Geo2rdr is picky about the type of height
        height_array = height_array.astype(np.double)

        geo2rdr_obj.set_geo_coordinate(np.radians(lon),
                                       np.radians(lat),
                                       1, 1,
                                       height_array)
        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los_x, los_y, los_z = geo2rdr_obj.get_los()
        loss[:, i] = los_x, los_y, los_z

        # get back the slant ranges
        slant_range = geo2rdr_obj.get_slant_range()
        slant_ranges[i] = slant_range

    los = loss * slant_ranges

    # Have to think about traversal order here. It's easy, though, since
    # in both orders xs come first, followed by all ys, followed by all
    # zs.
    return los.reshape((3,) + real_shape)


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


def gdal_open(fname, returnProj = False):
    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'
    ds = gdal.Open(fname, gdal.GA_ReadOnly)
    proj = ds.GetProjection()

    val = []
    for band in range(ds.RasterCount):
        b = ds.GetRasterBand(band + 1) # gdal counts from 1, not 0
        d = b.ReadAsArray()
        try:
            ndv = b.GetNoDataValue()
            d[d==ndv]=np.nan
        except:
            print('NoDataValue attempt failed*******')
            pass
        val.append(d)
        b = None
    ds = None

    if len(val) > 1:
        data = np.stack(val)
    else:
        data = val[0]

    if not returnProj:
        return data
    else:
        return data, proj



def pickle_dump(o, f):
    with open(f, 'wb') as fil:
        pickle.dump(o, fil)


def writeArrayToRaster(array, filename, fmt = 'ENVI'):
    # write a numpy array to a GDAL-readable raster
    import gdal
    import numpy as np
    array_shp = np.shape(array)
    dType = array.dtype
    if 'complex' in str(dType):
        dType = gdal.GDT_CFloat32
    elif 'float' in str(dType):
        dType = gdal.GDT_Float32
    else:
        dType = gdal.GDT_Byte

    driver = gdal.GetDriverByName(fmt)
    ds = driver.Create(filename, array_shp[1], array_shp[0],  1, dType)
    ds.GetRasterBand(1).WriteArray(array)
    ds = None


def calculategeoh(a, b, z, lnsp, ts, qs):
    geotoreturn = np.zeros_like(ts)
    pressurelvs = np.zeros_like(ts)

    Rd = 287.06

    z_h = 0

    # surface pressure
    sp = np.exp(lnsp)

    levelSize = len(ts)
    A = a
    B = b

    if len(a) != levelSize + 1 or len(b) != levelSize + 1:
        raise ValueError(
            f'I have here a model with {levelSize} levels, but parameters a '
            f'and b have lengths {len(a)} and {len(b)} respectively. Of '
            'course, these three numbers should be equal.')

    Ph_levplusone = A[levelSize] + (B[levelSize]*sp)

    # Integrate up into the atmosphere from lowest level
    for lev, t_level, q_level in zip(
            range(levelSize, 0, -1), ts[::-1], qs[::-1]):
        # lev is the level number 1-60, we need a corresponding index
        # into ts and qs
        ilevel = levelSize - lev

        # compute moist temperature
        t_level = t_level*(1 + 0.609133*q_level)

        # compute the pressures (on half-levels)
        Ph_lev = A[lev-1] + (B[lev-1] * sp)

        pressurelvs[ilevel] = Ph_lev

        if lev == 1:
            dlogP = np.log(Ph_levplusone/0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(Ph_levplusone/Ph_lev)
            dP = Ph_levplusone - Ph_lev
            alpha = 1 - ((Ph_lev/dP)*dlogP)

        TRd = t_level*Rd

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f = z_h + TRd*alpha

        # Geopotential (add in surface geopotential)
        geotoreturn[ilevel] = z_f + z

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h += TRd * dlogP

        Ph_levplusone = Ph_lev

    return geotoreturn, pressurelvs


def _find_svp(temp):
    """Calculate standard vapor presure."""
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
            * np.exp((17.502*(temp - 273.16))/(240.97 + temp - 273.16)))
    svpi = (6.1121
            * np.exp((22.587*(temp - 273.16))/(273.86 + temp - 273.16)))
    tempbound1 = 273.16  # 0
    tempbound2 = 250.16  # -23

    svp = svpw
    wgt = (temp - tempbound2)/(tempbound1 - tempbound2)
    svp = svpi + (svpw - svpi)*wgt**2
    ix_bound1 = temp > tempbound1
    svp[ix_bound1] = svpw[ix_bound1]
    ix_bound2 = temp < tempbound2
    svp[ix_bound2] = svpi[ix_bound2]

    return svp * 100
