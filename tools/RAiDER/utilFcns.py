"""Geodesy-related utility functions."""
import multiprocessing as mp
import os
import re

from datetime import datetime, timedelta

import h5py
import numpy as np
from numpy import ndarray
import pandas as pd
import pyproj
from pyproj import Transformer
from osgeo import gdal, osr
import progressbar

from RAiDER.constants import (
    _g0 as g0,
    R_EARTH_MAX as Rmax,
    R_EARTH_MIN as Rmin,
)
from RAiDER.logger import logger

gdal.UseExceptions()


def projectDelays(delay, inc):
    '''Project zenith delays to LOS'''
    return delay / cosd(inc)


# def floorish(val, frac):
#     '''Round a value to the lower fractional part'''
#     return val - (val % frac)


def sind(x):
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def lla2ecef(lat, lon, height):
    T = Transformer.from_crs(4326, 4978)

    return T.transform(lon, lat, height)


def enu2ecef(
    east: ndarray,
    north: ndarray,
    up: ndarray,
    lat0: ndarray,
    lon0: ndarray,
):
    """
    Parameters
    ----------
    e1 : float
        target east ENU coordinate (meters)
    n1 : float
        target north ENU coordinate (meters)
    u1 : float
        target up ENU coordinate (meters)
    Results
    -------
    u : float
    v : float
    w : float
    """
    t = cosd(lat0) * up - sind(lat0) * north
    w = sind(lat0) * up + cosd(lat0) * north

    u = cosd(lon0) * t - sind(lon0) * east
    v = sind(lon0) * t + cosd(lon0) * east

    return np.stack((u, v, w), axis=-1)


def ecef2enu(xyz, lat, lon, height):
    '''Convert ECEF xyz to ENU'''
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    t = cosd(lon) * x + sind(lon) * y

    e = -sind(lon) * x + cosd(lon) * y
    n = -sind(lat) * t + cosd(lat) * z
    u = cosd(lat) * t + sind(lat) * z
    return np.stack((e, n, u), axis=-1)


def gdal_extents(fname):
    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'
    try:
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
    except Exception:
        raise OSError('File {} could not be opened'.format(fname))

    # Check whether the file is georeferenced
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()
    if not proj or not gt:
        raise AttributeError('File {} does not contain geotransform information'.format(fname))

    xSize, ySize = ds.RasterXSize, ds.RasterYSize

    return [gt[0], gt[0] + (xSize - 1) * gt[1] + (ySize - 1) * gt[2], gt[3], gt[3] + (xSize - 1) * gt[4] + (ySize - 1) * gt[5]]


def gdal_open(fname, returnProj=False, userNDV=None):
    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'
    try:
        ds = gdal.Open(fname, gdal.GA_ReadOnly)
    except BaseException:  # TODO: Which error(s)?
        raise OSError('File {} could not be opened'.format(fname))
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()

    val = []
    for band in range(ds.RasterCount):
        b = ds.GetRasterBand(band + 1)  # gdal counts from 1, not 0
        data = b.ReadAsArray()
        if userNDV is not None:
            logger.debug('Using user-supplied NoDataValue')
            data[data == userNDV] = np.nan
        else:
            try:
                ndv = b.GetNoDataValue()
                data[data == ndv] = np.nan
            except BaseException:  # TODO: Which error(s)?
                logger.debug('NoDataValue attempt failed*******')
        val.append(data)
        b = None
    ds = None

    if len(val) > 1:
        data = np.stack(val)
    else:
        data = val[0]

    if not returnProj:
        return data
    else:
        return data, proj, gt


def writeResultsToHDF5(lats, lons, hgts, wet, hydro, filename, delayType=None):
    '''
    write a 1-D array to a NETCDF5 file
    '''
    if delayType is None:
        delayType = "Zenith"

    with h5py.File(filename, 'w') as f:
        f['lat'] = lats
        f['lon'] = lons
        f['hgts'] = hgts
        f['wetDelay'] = wet
        f['hydroDelay'] = hydro
        f['wetDelayUnit'] = "m"
        f['hydroDelayUnit'] = "m"
        f['hgtsUnit'] = "m"
        f.attrs['DelayType'] = delayType


def writeArrayToRaster(array, filename, noDataValue=0., fmt='ENVI', proj=None, gt=None):
    '''
    write a numpy array to a GDAL-readable raster
    '''
    array_shp = np.shape(array)
    if array.ndim != 2:
        raise RuntimeError('writeArrayToRaster: cannot write an array of shape {} to a raster image'.format(array_shp))
    dType = array.dtype
    if 'complex' in str(dType):
        dType = gdal.GDT_CFloat32
    elif 'float' in str(dType):
        dType = gdal.GDT_Float32
    else:
        dType = gdal.GDT_Byte

    driver = gdal.GetDriverByName(fmt)
    ds = driver.Create(filename, array_shp[1], array_shp[0], 1, dType)
    if proj is not None:
        ds.SetProjection(proj)
    if gt is not None:
        ds.SetGeoTransform(gt)
    b1 = ds.GetRasterBand(1)
    b1.WriteArray(array)
    b1.SetNoDataValue(noDataValue)
    ds = None
    b1 = None


def writeArrayToFile(lats, lons, array, filename, noDataValue=-9999):
    '''
    Write a single-dim array of values to a file
    '''
    array[np.isnan(array)] = noDataValue
    with open(filename, 'w') as f:
        f.write('Lat,Lon,Hgt_m\n')
        for lat, lon, height in zip(lats, lons, array):
            f.write('{},{},{}\n'.format(lat, lon, height))


def round_date(date, precision):
    # First try rounding up
    # Timedelta since the beginning of time
    datedelta = datetime.min - date
    # Round that timedelta to the specified precision
    rem = datedelta % precision
    # Add back to get date rounded up
    round_up = date + rem

    # Next try rounding down
    datedelta = date - datetime.min
    rem = datedelta % precision
    round_down = date - rem

    # It's not the most efficient to calculate both and then choose, but
    # it's clear, and performance isn't critical here.
    up_diff = round_up - date
    down_diff = date - round_down

    return round_up if up_diff < down_diff else round_down


def _least_nonzero(a):
    """Fill in a flat array with the first non-nan value in the last dimension.

    Useful for interpolation below the bottom of the weather model.
    """
    mgrid_index = tuple(slice(None, d) for d in a.shape[:-1])
    return a[tuple(np.mgrid[mgrid_index]) + ((~np.isnan(a)).argmax(-1),)]


def robmin(a):
    '''
    Get the minimum of an array, accounting for empty lists
    '''
    try:
        return np.nanmin(a)
    except ValueError:
        return 'N/A'


def robmax(a):
    '''
    Get the minimum of an array, accounting for empty lists
    '''
    try:
        return np.nanmax(a)
    except ValueError:
        return 'N/A'


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
    return np.sqrt(1 / (((cosd(lats)**2) / Rmax**2) + ((sind(lats)**2) / Rmin**2)))


def _geo_to_ht(lats, hts):
    """Convert geopotential height to altitude."""
    # Convert geopotential to geometric height. This comes straight from
    # TRAIN
    # Map of g with latitude (I'm skeptical of this equation - Ray)
    g_ll = _get_g_ll(lats)
    Re = _get_Re(lats)

    # Calculate Geometric Height, h
    h = (hts * Re) / (g_ll / g0 * Re - hts)
    # from metpy
    # return (geopotential * Re) / (g0 * Re - geopotential)

    return h


def padLower(invar):
    '''
    add a layer of data below the lowest current z-level at height zmin
    '''
    new_var = _least_nonzero(invar)
    return np.concatenate((new_var[:, :, np.newaxis], invar), axis=2)


# def checkShapes(los, lats, lons, hts):
#     '''
#     Make sure that by the time the code reaches here, we have a
#     consistent set of line-of-sight and position data.
#     '''
#     from RAiDER.losreader import Zenith
#     test1 = hts.shape == lats.shape == lons.shape
#     try:
#         test2 = los.shape[:-1] == hts.shape
#     except AttributeError:
#         test2 = los is Zenith

#     if not test1 and test2:
#         raise ValueError(
#             'I need lats, lons, heights, and los to all be the same shape. ' +
#             'lats had shape {}, lons had shape {}, '.format(lats.shape, lons.shape) +
#             'heights had shape {}, and los was not Zenith'.format(hts.shape))


def checkLOS(los, Npts):
    '''
    Check that los is either:
       (1) Zenith,
       (2) a set of scalar values of the same size as the number
           of points, which represent the projection value), or
       (3) a set of vectors, same number as the number of points.
     '''
    from RAiDER.losreader import Zenith

    # los is a bunch of vectors or Zenith
    if los is not Zenith:
        los = los.reshape(-1, 3)

    if los is not Zenith and los.shape[0] != Npts:
        raise RuntimeError('Found {} line-of-sight values and only {} points'
                           .format(los.shape[0], Npts))
    return los


def read_hgt_file(filename):
    '''
    Read height data from a comma-delimited file
    '''
    data = pd.read_csv(filename)
    hgts = data['Hgt_m'].values
    return hgts


def roundTime(dt, roundTo=60):
    '''
    Round a datetime object to any time lapse in seconds
    dt: datetime.datetime object
    roundTo: Closest number of seconds to round to, default 1 minute.
    Source: https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object/10854034#10854034
    '''
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                wetFilename, hydroFilename=None, zlevels=None, delayType=None,
                outformat=None, proj=None, gt=None, ndv=0.):
    '''
    Write the delay numpy arrays to files in the format specified
    '''

    # Need to consistently handle noDataValues
    wetDelay[np.isnan(wetDelay)] = ndv
    hydroDelay[np.isnan(hydroDelay)] = ndv

    # Do different things, depending on the type of input
    if flag == 'station_file':
        try:
            df = pd.read_csv(wetFilename)
        except ValueError:
            wetFilename = wetFilename[0]
            df = pd.read_csv(wetFilename)

        df['wetDelay'] = wetDelay
        df['hydroDelay'] = hydroDelay
        df['totalDelay'] = wetDelay + hydroDelay
        df.to_csv(wetFilename, index=False)

    elif outformat == 'hdf5':
        writeResultsToHDF5(
            lats,
            lons,
            zlevels,
            wetDelay,
            hydroDelay,
            wetFilename,
            delayType=delayType
        )
    else:
        writeArrayToRaster(
            wetDelay,
            wetFilename,
            noDataValue=ndv,
            fmt=outformat,
            proj=proj,
            gt=gt
        )
        writeArrayToRaster(
            hydroDelay,
            hydroFilename,
            noDataValue=ndv,
            fmt=outformat,
            proj=proj,
            gt=gt
        )


def getTimeFromFile(filename):
    '''
    Parse a filename to get a date-time
    '''
    fmt = '%Y_%m_%d_T%H_%M_%S'
    p = re.compile(r'\d{4}_\d{2}_\d{2}_T\d{2}_\d{2}_\d{2}')
    try:
        out = p.search(filename).group()
        return datetime.strptime(out, fmt)
    except BaseException:  # TODO: Which error(s)?
        raise RuntimeError('The filename for {} does not include a datetime in the correct format'.format(filename))


def writePnts2HDF5(lats, lons, hgts, los, lengths, out_name='testx.h5', chunkSize=None, noDataValue=0., epsg=4326):
    '''
    Write query points to an HDF5 file for storage and access
    '''
    projname = 'projection'

    # converts from WGS84 geodetic to WGS84 geocentric
    t = Transformer.from_crs(epsg, 4978, always_xy=True)

    checkLOS(los, np.prod(lats.shape))
    in_shape = lats.shape

    # create directory if needed
    os.makedirs(os.path.abspath(os.path.dirname(out_name)), exist_ok=True)

    # Set up the chunking
    if chunkSize is None:
        chunkSize = getChunkSize(in_shape)

    with h5py.File(out_name, 'w') as f:
        f.attrs['Conventions'] = np.string_("CF-1.8")

        x = f.create_dataset('lon', data=lons, chunks=chunkSize, fillvalue=noDataValue)
        y = f.create_dataset('lat', data=lats, chunks=chunkSize, fillvalue=noDataValue)
        z = f.create_dataset('hgt', data=hgts, chunks=chunkSize, fillvalue=noDataValue)
        los = f.create_dataset(
            'LOS',
            data=los,
            chunks=chunkSize + (3,),
            fillvalue=noDataValue
        )
        lengths = f.create_dataset(
            'Rays_len',
            data=lengths,
            chunks=x.chunks,
            fillvalue=noDataValue
        )
        sp_data = np.stack(t.transform(lons, lats, hgts), axis=-1).astype(np.float64)
        sp = f.create_dataset(
            'Rays_SP',
            data=sp_data,
            chunks=chunkSize + (3,),
            fillvalue=noDataValue
        )

        x.attrs['Shape'] = in_shape
        y.attrs['Shape'] = in_shape
        z.attrs['Shape'] = in_shape
        los.attrs['Shape'] = in_shape + (3,)
        lengths.attrs['Shape'] = in_shape
        lengths.attrs['Units'] = 'm'
        sp.attrs['Shape'] = in_shape + (3,)
        f.attrs['ChunkSize'] = chunkSize
        f.attrs['NoDataValue'] = noDataValue

        # CF 1.8 Convention stuff
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        projds = f.create_dataset(projname, (), dtype='i')
        projds[()] = epsg

        # WGS84 ellipsoid
        projds.attrs['semi_major_axis'] = 6378137.0
        projds.attrs['inverse_flattening'] = 298.257223563
        projds.attrs['ellipsoid'] = np.string_("WGS84")
        projds.attrs['epsg_code'] = epsg
        projds.attrs['spatial_ref'] = np.string_(srs.ExportToWkt())

        # Geodetic latitude / longitude
        if epsg == 4326:
            # Set up grid mapping
            projds.attrs['grid_mapping_name'] = np.string_('latitude_longitude')
            projds.attrs['longitude_of_prime_meridian'] = 0.0

            x.attrs['standard_name'] = np.string_("longitude")
            x.attrs['units'] = np.string_("degrees_east")
            y.attrs['standard_name'] = np.string_("latitude")
            y.attrs['units'] = np.string_("degrees_north")
            z.attrs['standard_name'] = np.string_("height")
            z.attrs['units'] = np.string_("m")
        else:
            raise NotImplementedError

        los.attrs['grid_mapping'] = np.string_(projname)
        sp.attrs['grid_mapping'] = np.string_(projname)
        lengths.attrs['grid_mapping'] = np.string_(projname)

        f.attrs['NumRays'] = len(x)
        f['Rays_len'].attrs['MaxLen'] = np.nanmax(lengths)


# Part of the following UTM and WGS84 converter is borrowed from https://gist.github.com/twpayne/4409500
# Credits go to Tom Payne

_projections = {}


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


def letter(coordinates):
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


def project(coordinates, z=None, l=None):
    if z is None:
        z = zone(coordinates)
    if l is None:
        l = letter(coordinates)
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _projections[z](coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, l, x, y


# def unproject(z, l, x, y):
#     if z not in _projections:
#         _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
#     if l < 'N':
#         y -= 10000000
#     lng, lat = _projections[z](x, y, inverse=True)
#     return (lng, lat)


def WGS84_to_UTM(lon, lat, common_center=False):
    shp = lat.shape
    lon = np.ravel(lon)
    lat = np.ravel(lat)
    if common_center:
        lon0 = np.median(lon)
        lat0 = np.median(lat)
        z0, l0, x0, y0 = project((lon0, lat0))
    Z = lon.copy()
    L = np.zeros(lon.shape, dtype='<U1')
    X = lon.copy()
    Y = lon.copy()
    for ind in range(lon.__len__()):
        longitude = lon[ind]
        latitude = lat[ind]
        if common_center:
            z, l, x, y = project((longitude, latitude), z0, l0)
        else:
            z, l, x, y = project((longitude, latitude))
        Z[ind] = z
        L[ind] = l
        X[ind] = x
        Y[ind] = y
    return np.reshape(Z, shp), np.reshape(L, shp), np.reshape(X, shp), np.reshape(Y, shp)


# def UTM_to_WGS84(z, l, x, y):
#     shp = x.shape
#     z = np.ravel(z)
#     l = np.ravel(l)
#     x = np.ravel(x)
#     y = np.ravel(y)
#     lat = x.copy()
#     lon = x.copy()
#     for ind in range(z.__len__()):
#         zz = z[ind]
#         ll = l[ind]
#         xx = x[ind]
#         yy = y[ind]
#         coordinates = unproject(zz, ll, xx, yy)
#         lat[ind] = coordinates[1]
#         lon[ind] = coordinates[0]
#     return np.reshape(lon, shp), np.reshape(lat, shp)


def requests_retry_session(retries=10, session=None):
    """ https://www.peterbe.com/plog/best-practice-with-retries-with-requests """
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    # add a retry strategy; https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
    session = session or requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries,
                  backoff_factor=0.3, status_forcelist=list(range(429, 505)))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def writeWeatherVars2NETCDF4(self, lat, lon, h, q, p, t, out_name=None, NoDataValue=None, chunk=(1, 91, 144), mapping_name='WGS84'):
    '''
    By calling the abstract/modular netcdf writer (RAiDER.utilFcns.write2NETCDF4core), write the OpenDAP/PyDAP-retrieved weather model data (GMAO and MERRA-2) to a NETCDF4 file
    that can be accessed by external programs.

    The point of doing this is to alleviate some of the memory load of keeping
    the full model in memory and make it easier to scale up the program.
    '''

    import netCDF4

    if out_name is None:
        out_name = os.path.join(
            os.getcwd() + '/weather_files',
            self._Name + datetime.strftime(
                self._time, '_%Y_%m_%d_T%H_%M_%S'
            ) + '.nc'
        )

    if NoDataValue is None:
        NoDataValue = -9999.

    self._time = getTimeFromFile(out_name)

    dimidZ, dimidY, dimidX = t.shape
    chunk_lines_Y = np.min([chunk[1], dimidY])
    chunk_lines_X = np.min([chunk[2], dimidX])
    ChunkSize = [1, chunk_lines_Y, chunk_lines_X]

    nc_outfile = netCDF4.Dataset(out_name, 'w', clobber=True, format='NETCDF4')
    nc_outfile.setncattr('Conventions', 'CF-1.6')
    nc_outfile.setncattr('datetime', datetime.strftime(self._time, "%Y_%m_%dT%H_%M_%S"))
    nc_outfile.setncattr('date_created', datetime.now().strftime("%Y_%m_%dT%H_%M_%S"))
    title = self._Name + ' weather model data'
    nc_outfile.setncattr('title', title)

    tran = [lon[0], lon[1] - lon[0], 0.0, lat[0], 0.0, lat[1] - lat[0]]

    dimension_dict = {
        'x': {
            'varname': 'x',
            'datatype': np.dtype('float64'),
            'dimensions': ('x'),
            'length': dimidX,
            'FillValue': None,
            'standard_name': 'longitude',
            'description': 'longitude',
            'dataset': lon,
            'units': 'degrees_east'
        },
        'y': {
            'varname': 'y',
            'datatype': np.dtype('float64'),
            'dimensions': ('y'),
            'length': dimidY,
            'FillValue': None,
            'standard_name': 'latitude',
            'description': 'latitude',
            'dataset': lat,
            'units': 'degrees_north'
        },
        'z': {
            'varname': 'z',
            'datatype': np.dtype('float32'),
            'dimensions': ('z'),
            'length': dimidZ,
            'FillValue': None,
            'standard_name': 'model_layers',
            'description': 'model layers',
            'dataset': np.arange(dimidZ),
            'units': 'layer'
        }
    }

    dataset_dict = {
        'h': {
            'varname': 'H',
            'datatype': np.dtype('float32'),
            'dimensions': ('z', 'y', 'x'),
            'grid_mapping': mapping_name,
            'FillValue': NoDataValue,
            'ChunkSize': ChunkSize,
            'standard_name': 'mid_layer_heights',
            'description': 'mid layer heights',
            'dataset': h,
            'units': 'm'
        },
        'q': {
            'varname': 'QV',
            'datatype': np.dtype('float32'),
            'dimensions': ('z', 'y', 'x'),
            'grid_mapping': mapping_name,
            'FillValue': NoDataValue,
            'ChunkSize': ChunkSize,
            'standard_name': 'specific_humidity',
            'description': 'specific humidity',
            'dataset': q,
            'units': 'kg kg-1'
        },
        'p': {
            'varname': 'PL',
            'datatype': np.dtype('float32'),
            'dimensions': ('z', 'y', 'x'),
            'grid_mapping': mapping_name,
            'FillValue': NoDataValue,
            'ChunkSize': ChunkSize,
            'standard_name': 'mid_level_pressure',
            'description': 'mid level pressure',
            'dataset': p,
            'units': 'Pa'
        },
        't': {
            'varname': 'T',
            'datatype': np.dtype('float32'),
            'dimensions': ('z', 'y', 'x'),
            'grid_mapping': mapping_name,
            'FillValue': NoDataValue,
            'ChunkSize': ChunkSize,
            'standard_name': 'air_temperature',
            'description': 'air temperature',
            'dataset': t,
            'units': 'K'
        }
    }

    nc_outfile = write2NETCDF4core(nc_outfile, dimension_dict, dataset_dict, tran, mapping_name='WGS84')

    nc_outfile.sync()  # flush data to disk
    nc_outfile.close()


def write2NETCDF4core(nc_outfile, dimension_dict, dataset_dict, tran, mapping_name='WGS84'):
    '''
    The abstract/modular netcdf writer that can be called by a wrapper function to write data to a NETCDF4 file
    that can be accessed by external programs.

    The point of doing this is to alleviate some of the memory load of keeping
    the full model in memory and make it easier to scale up the program.
    '''
    from osgeo import osr

    if mapping_name == 'WGS84':

        epsg = 4326
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)

        grid_mapping = 'WGS84'  # need to set this as an attribute for the image variables
        datatype = np.dtype('S1')
        dimensions = ()

        var = nc_outfile.createVariable(
            mapping_name,
            datatype,
            dimensions,
            fill_value=None
        )
        # variable made, now add attributes

        var.setncattr('grid_mapping_name', grid_mapping)
        var.setncattr('straight_vertical_longitude_from_pole', srs.GetProjParm('central_meridian'))
        var.setncattr('false_easting', srs.GetProjParm('false_easting'))
        var.setncattr('false_northing', srs.GetProjParm('false_northing'))
        var.setncattr('latitude_of_projection_origin', np.sign(srs.GetProjParm('latitude_of_origin')) * 90.0)
        var.setncattr('latitude_of_origin', srs.GetProjParm('latitude_of_origin'))
        var.setncattr('semi_major_axis', float(srs.GetAttrValue('GEOGCS|SPHEROID', 1)))
        var.setncattr('scale_factor_at_projection_origin', 1)
        var.setncattr('inverse_flattening', float(srs.GetAttrValue('GEOGCS|SPHEROID', 2)))
        var.setncattr('spatial_ref', srs.ExportToWkt())
        var.setncattr('spatial_proj4', srs.ExportToProj4())
        var.setncattr('spatial_epsg', epsg)
        var.setncattr('GeoTransform', ' '.join(str(x) for x in tran))  # note this has pixel size in it - set  explicitly above

    else:
        raise Exception('Grid mapping name not supported; currently, only WGS84 (EPSG: 4326) is supported!')

    for dim in dimension_dict:
        nc_outfile.createDimension(dim, dimension_dict[dim]['length'])
        varname = dimension_dict[dim]['varname']
        datatype = dimension_dict[dim]['datatype']
        dimensions = dimension_dict[dim]['dimensions']
        FillValue = dimension_dict[dim]['FillValue']
        var = nc_outfile.createVariable(varname, datatype, dimensions, fill_value=FillValue)
        var.setncattr('standard_name', dimension_dict[dim]['standard_name'])
        var.setncattr('description', dimension_dict[dim]['description'])
        var.setncattr('units', dimension_dict[dim]['units'])
        var[:] = dimension_dict[dim]['dataset'].astype(dimension_dict[dim]['datatype'])

    for data in dataset_dict:
        varname = dataset_dict[data]['varname']
        datatype = dataset_dict[data]['datatype']
        dimensions = dataset_dict[data]['dimensions']
        FillValue = dataset_dict[data]['FillValue']
        ChunkSize = dataset_dict[data]['ChunkSize']
        var = nc_outfile.createVariable(
            varname,
            datatype,
            dimensions,
            fill_value=FillValue,
            zlib=True,
            complevel=2,
            shuffle=True,
            chunksizes=ChunkSize
        )
        var.setncattr('grid_mapping', dataset_dict[data]['grid_mapping'])
        var.setncattr('standard_name', dataset_dict[data]['standard_name'])
        var.setncattr('description', dataset_dict[data]['description'])
        if 'units' in dataset_dict[data]:
            var.setncattr('units', dataset_dict[data]['units'])

        ndmask = np.isnan(dataset_dict[data]['dataset'])
        dataset_dict[data]['dataset'][ndmask] = FillValue

        var[:] = dataset_dict[data]['dataset'].astype(datatype)

    return nc_outfile


# def convertLons(inLons):
#     '''Convert lons from 0-360 to -180-180'''
#     mask = inLons > 180
#     outLons = inLons
#     outLons[mask] = outLons[mask] - 360
#     return outLons


def read_NCMR_loginInfo(filepath=None):
    from pathlib import Path

    if filepath is None:
        filepath = str(Path.home()) + '/.ncmrlogin'

    f = open(filepath, 'r')
    lines = f.readlines()
    url = lines[0].strip().split(': ')[1]
    username = lines[1].strip().split(': ')[1]
    password = lines[2].strip().split(': ')[1]

    return url, username, password


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def getChunkSize(in_shape):
    '''Create a reasonable chunk size'''
    minChunkSize = 100
    maxChunkSize = 1000
    cpu_count = mp.cpu_count()
    chunkSize = tuple(
        max(
            min(maxChunkSize, s // cpu_count),
            min(s, minChunkSize)
        ) for s in in_shape
    )
    return chunkSize
