""" General function related to file I/O """
import h5py
import importlib
import os
import pyproj
import re
import requests

import multiprocessing as mp
import numpy as np
import pandas as pd

from datetime import datetime
from osgeo import gdal, osr
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

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




def writeArrayToFile(lats, lons, array, filename, noDataValue=-9999):
    '''
    Write a single-dim array of values to a file
    '''
    array[np.isnan(array)] = noDataValue
    with open(filename, 'w') as f:
        f.write('Lat,Lon,Hgt_m\n')
        for l, L, a in zip(lats, lons, array):
            f.write('{},{},{}\n'.format(l, L, a))


def makeDelayFileNames(time, los, outformat, weather_model_name, out):
    '''
    return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(time(0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    '''
    format_string = "{model_name}_{{}}_{time}{los}.{ext}".format(
        model_name=weather_model_name,
        time=time.strftime("%H_%M_%S_") if time is not None else "",
        los="ztd" if los is None else "std",
        ext=outformat
    )
    hydroname, wetname = (
        format_string.format(dtyp) for dtyp in ('hydro', 'wet')
    )

    hydro_file_name = os.path.join(out, hydroname)
    wet_file_name = os.path.join(out, wetname)
    return wet_file_name, hydro_file_name


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


def writePnts2HDF5(lats, lons, hgts, los, outName='testx.h5', chunkSize=None, noDataValue=0.):
    '''
    Write query points to an HDF5 file for storage and access
    '''
    epsg = 4326
    projname = 'projection'

    checkLOS(los, np.prod(lats.shape))
    in_shape = lats.shape

    # create directory if needed
    os.makedirs(os.path.abspath(os.path.dirname(outName)), exist_ok=True)

    if chunkSize is None:
        minChunkSize = 100
        maxChunkSize = 10000
        cpu_count = mp.cpu_count()
        chunkSize = tuple(max(min(maxChunkSize, s // cpu_count), min(s, minChunkSize)) for s in in_shape)

    with h5py.File(outName, 'w') as f:
        f.attrs['Conventions'] = np.string_("CF-1.8")

        x = f.create_dataset('lon', data=lons, chunks=chunkSize, fillvalue=noDataValue)
        y = f.create_dataset('lat', data=lats, chunks=chunkSize, fillvalue=noDataValue)
        z = f.create_dataset('hgt', data=hgts, chunks=chunkSize, fillvalue=noDataValue)
        los = f.create_dataset('LOS', data=los, chunks=chunkSize + (3,), fillvalue=noDataValue)
        x.attrs['Shape'] = in_shape
        y.attrs['Shape'] = in_shape
        z.attrs['Shape'] = in_shape
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
            raise NotImplemented

        start_positions = f.create_dataset('Rays_SP', in_shape + (3,), chunks=los.chunks, dtype='<f8', fillvalue=noDataValue)
        lengths = f.create_dataset('Rays_len', in_shape, chunks=x.chunks, dtype='<f8', fillvalue=noDataValue)
        scaled_look_vecs = f.create_dataset('Rays_SLV', in_shape + (3,), chunks=los.chunks, dtype='<f8', fillvalue=noDataValue)

        los.attrs['grid_mapping'] = np.string_(projname)
        start_positions.attrs['grid_mapping'] = np.string_(projname)
        lengths.attrs['grid_mapping'] = np.string_(projname)
        scaled_look_vecs.attrs['grid_mapping'] = np.string_(projname)

        f.attrs['NumRays'] = len(x)


def writeWeatherVars2HDF5(lat, lon, x, y, z, q, p, t, proj, outName=None):
    '''
    Write the OpenDAP/PyDAP-retrieved weather model data (GMAO and MERRA-2) to an HDF5 file
    that can be accessed by external programs.

    The point of doing this is to alleviate some of the memory load of keeping
    the full model in memory and make it easier to scale up the program.
    '''

    if outName is None:
        outName = os.path.join(
            os.getcwd()+'/weather_files',
            self._Name + datetime.strftime(
                self._time, '_%Y_%m_%d_T%H_%M_%S'
            ) + '.h5'
        )

    with h5py.File(outName, 'w') as f:
        lon = f.create_dataset('lons', data=lon.astype(np.float64))
        lat = f.create_dataset('lats', data=lat.astype(np.float64))

        X = f.create_dataset('x', data=x)
        Y = f.create_dataset('y', data=y)
        Z = f.create_dataset('z', data=z)

        Q = f.create_dataset('q', data=q)
        P = f.create_dataset('p', data=p)
        T = f.create_dataset('t', data=t)

        f.create_dataset('Projection', data=proj.to_json())

def write(varDict,outName,fmt='h5'):
    #lat, lon, x, y, z, q, p, t, proj
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
        writeVars2HDF5(varDict, outName)
    else:
        writeVars2NETCDF4(varDict, outName)

def writeVars2HDF5(varDict, outName):
    ''' Write variables to an HDF5 file '''
    with h5py.File(outName, 'w') as f:
        for var in varDict.keys():
            v = f.create_dataset(
                    var, 
                    data = varDict[var]['data']
                )

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
