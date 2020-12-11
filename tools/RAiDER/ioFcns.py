""" General function related to file I/O """
import h5py
import importlib
import os
import pyproj
import re
import requests

import numpy as np
import pandas as pd
import xarray as xr

from datetime import datetime
from osgeo import gdal, osr
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from RAiDER.geometry import checkLOS

gdal.UseExceptions()


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
    except:
        raise OSError('File {} could not be opened'.format(fname))
    proj = ds.GetProjection()
    gt = ds.GetGeoTransform()

    val = []
    for band in range(ds.RasterCount):
        b = ds.GetRasterBand(band + 1)  # gdal counts from 1, not 0
        data = b.ReadAsArray()
        if userNDV is not None:
            data[data == userNDV] = np.nan
        else:
            try:
                ndv = b.GetNoDataValue()
                data[data == ndv] = np.nan
            except:
                pass
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


def writeArrayToFile(
        lats, 
        lons, 
        array, 
        filename, 
        noDataValue=-9999
    ):
    '''
    Write a single-dim array of values to a file
    '''
    array[np.isnan(array)] = noDataValue
    with open(filename, 'w') as f:
        f.write('Lat,Lon,Hgt_m\n')
        for l, L, a in zip(lats, lons, array):
            f.write('{},{},{}\n'.format(l, L, a))




def make_weather_model_filename(name, time, ll_bounds):
    if ll_bounds[0] < 0:
        S = 'S'
    else:
        S = 'N'
    if ll_bounds[1] < 0:
        N = 'S'
    else:
        N = 'N'
    if ll_bounds[2] < 0:
        W = 'W'
    else:
        W = 'E'
    if ll_bounds[3] < 0:
        E = 'W'
    else:
        E = 'E'
    return '{}_{}_{}{}_{}{}_{}{}_{}{}.h5'.format(
        name, time.strftime("%Y-%m-%dT%H_%M_%S"), np.abs(ll_bounds[0]), S, np.abs(ll_bounds[1]), N, np.abs(ll_bounds[2]), W, np.abs(ll_bounds[3]), E
    )


def read_hgt_file(filename):
    '''
    Read height data from a comma-delimited file
    '''
    data = pd.read_csv(filename)
    hgts = data['Hgt_m'].values
    return hgts


def writeDelays(
        flag, 
        wetDelay, 
        hydroDelay, 
        lats, 
        lons,
        wetFilename, 
        hydroFilename=None, 
        zlevels=None, 
        delayType=None,
        outformat=None, 
        proj=None, 
        gt=None, 
        ndv=0.
    ):
    '''
    Write the delay numpy arrays to files in the format 
    specified
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

        # quick check for consistency
        assert(np.all(np.abs(lats - df['Lat']) < 0.01))

        df['wetDelay'] = wetDelay
        df['hydroDelay'] = hydroDelay
        df['totalDelay'] = wetDelay + hydroDelay
        df.to_csv(wetFilename, index=False)

    elif outformat == 'hdf5':
        writeVars2HDF5(
            {
                'lats': lats,
                'lons': lons,
                'zlevels': zlevels,
                'wetDelay': wetDelay,
                'hydroDelay': hydroDelay,
            },
            outName = wetFilename, 
            attrs = {
                'DelayType': delayType
            },
            NoDataValue=ndv
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


def writeArrayToRaster(
        array, 
        filename, 
        noDataValue=0., 
        fmt='ENVI', 
        proj=None, 
        gt=None
    ):
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


def getTimeFromFile(filename):
    '''
    Parse a filename to get a date-time
    '''
    fmt = '%Y_%m_%d_T%H_%M_%S'
    p = re.compile(r'\d{4}_\d{2}_\d{2}_T\d{2}_\d{2}_\d{2}')
    try:
        out = p.search(filename).group()
        return datetime.strptime(out, fmt)
    except:
        raise RuntimeError('The filename for {} does not include a datetime in the correct format'.format(filename))


def write(
        varDict,
        outName, 
        attrs = None, 
        chunkSize = None, 
        NoDataValue = None, 
        fmt='h5'
    ):
    '''
    Write variables to a file. 
    '''
    _allowed_fmts = ['h5', 'nc4']

    if fmt not in _allowed_fmts:
        raise ValueError(
                'Format {} not allowed, must be h5 or nc4'
                .format(fmt)
            )

    if fmt == 'h5':
        writeVars2HDF5(
                varDict, 
                outName,
                attrs = attrs,
                chunkSize = chunkSize,
                NoDataValue = NoDataValue
            )
    else:
        writeVars2NETCDF4(varDict, outName)


def writeVars2HDF5(
        varDict, 
        outName, 
        attrs=None, 
        chunkSize=None,
        NoDataValue=None
    ):
    ''' 
    Write variables to an HDF5 file 

    Parameters
    ----------
    varDict     - Dict of dicts containing variables to write
    outName     - Filename to write to
    attrs       - Dict of attributes to write to file
    chunkSize   - tuple to use for writing chunks
    NoDataValue - A single NoDataValue for all variables

    Example
    -------
    >>> d1 = np.random.randn(10,2)
    >>> fname = 'test.h5'
    >>> attrs = {'attribute': 'test'}
    >>> d = {
    >>>     'var1': {
    >>>         'data': d1, 
    >>>         'attrs': {
    >>>             'NoDataValue': -999., 
    >>>             'proj': 'wgs-84'
    >>>         }
    >>>     }
    >>> }
    >>> writeVars2HDF5(d, fname, attrs)
    '''
    with h5py.File(outName, 'w') as f:

        # Write the attributes dict to the file
        if attrs is not None:
            for var in attrs.keys():
                f.attrs[var] = attrs[var]

        # Write each var dict to the file
        for var in varDict.keys():
            tmp = f.create_dataset(
                    var, 
                    data = varDict[var]['data'],
                    chunks = chunkSize,
                    fillvalue = NoDataValue,
                )
            try:
                for attr, value in varDict[var]['attrs'].items():
                    tmp.attrs[attr] = value
            except KeyError:
                pass


def writeVars2NETCDF4(
        varDict, 
        outName,
        attrs=None, 
        NoDataValue=None
    ):
    '''
    Write variables to NETCDF4. 

    Parameters
    ----------
    varDict     - A dict of dicts. 
    
    Keys should be variable names, values should be dicts. 
    Each dict should have the following form:
        "coords": A dict of coordinate variable names, values, and attributes
        "attrs": A dict of attributes for the variable
        "dims": dimension name
        "data": The multi-dimensional data itself
        "name": The name of the variable

    outName     - Filename to write to
    attrs       - Dict of attributes to write to file
    chunkSize   - tuple to use for writing chunks
    NoDataValue - A single NoDataValue for all variables

    Example
    -------
    >>> d1 = np.random.randn(10,2)
    >>> fname = 'test.h5'
    >>> attrs = {'attribute': 'test'}
    >>> d = {
    >>>     'var1': {
    >>>         'data': d1, 
    >>>         'attrs': {
    >>>             'NoDataValue': -999., 
    >>>             'proj': 'wgs-84'
    >>>         }
    >>>     }
    >>> }
    >>> writeVars2NETCDF(d, fname, attrs)
    '''
    ds = xr.Dataset.from_dict(varDict)
    ds.attrs = attrs
    



def requests_retry_session(retries=10, session=None):
    """ 
    https://www.peterbe.com/plog/best-practice-with-retries-with-requests 
    """
    # add a retry strategy; https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
    session = session or requests.Session()
    retry   = Retry(total=retries, read=retries, connect=retries,
                    backoff_factor=0.3, status_forcelist=list(range(429, 505)))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

