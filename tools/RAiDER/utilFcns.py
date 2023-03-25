"""Geodesy-related utility functions."""
import multiprocessing as mp
import os
import re
import xarray

from datetime import datetime, timedelta

import h5py
import xarray as xr
import numpy as np
from numpy import ndarray
import pandas as pd
import pyproj
from pyproj import Transformer
import progressbar
import rasterio

from RAiDER.constants import (
    _g0 as g0,
    _g1 as G1,
    R_EARTH_MAX_WGS84 as Rmax,
    R_EARTH_MIN_WGS84 as Rmin,
    _THRESHOLD_SECONDS,
)
from RAiDER.logger import logger


pbar = None


def projectDelays(delay, inc):
    '''Project zenith delays to LOS'''
    return delay / cosd(inc)


def floorish(val, frac):
    '''Round a value to the lower fractional part'''
    return val - (val % frac)


def sind(x):
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def lla2ecef(lat, lon, height):
    T = Transformer.from_crs(4326, 4978, always_xy=True)

    return T.transform(lon, lat, height)


def ecef2lla(x, y, z):
    T = Transformer.from_crs(4978, 4326, always_xy=True)

    return T.transform(x, y, z)


def enu2ecef(
    east: ndarray,
    north: ndarray,
    up: ndarray,
    lat0: ndarray,
    lon0: ndarray,
    h0: ndarray,
):
    """
    Args:
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


def rio_profile(fname):
    ## need to access subdataset directly
    if os.path.basename(fname).startswith('S1-GUNW'):
        fname = os.path.join(f'NETCDF:"{fname}":science/grids/data/unwrappedPhase')
        with rasterio.open(fname) as src:
            profile = src.profile

    elif os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'

    with rasterio.open(fname) as src:
        profile = src.profile
        # if 'S1-GUNW' in fname:
            # profile['length'] = profile['width']
            # profile['width']  = profile['height']

    if profile["crs"] is None:
        raise AttributeError(
            f"{fname} does not contain geotransform information"
        )

    return profile


def rio_extents(profile):
    """ Get a bounding box in SNWE from a rasterio profile """
    gt = profile["transform"].to_gdal()
    xSize = profile["width"]
    ySize = profile["height"]

    if profile["crs"] is None or not gt:
        raise AttributeError('Profile does not contain geotransform information')
    W, E = gt[0], gt[0] + (xSize - 1) * gt[1] + (ySize - 1) * gt[2]
    N, S = gt[3], gt[3] + (xSize - 1) * gt[4] + (ySize - 1) * gt[5]
    return S, N, W, E


def rio_open(fname, returnProj=False, userNDV=None, band=None):
    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'

    with rasterio.open(fname) as src:
        profile = src.profile

        # For all bands
        nodata = src.nodatavals

        # If user requests a band
        if band is not None:
            ndv = nodata[band - 1]
            data = src.read(band).squeeze()
            nodataToNan(data, [userNDV, nodata[band - 1]])

        else:
            data = src.read().squeeze()
            if data.ndim > 2:
                for bnd in range(data.shape[0]):
                    val = data[bnd, ...]
                    nodataToNan(val, [userNDV, nodata[bnd]])
            else:
                nodataToNan(data, list(nodata) + [userNDV])


        if data.ndim > 2:
            dlist = []
            for k in range(data.shape[0]):
                dlist.append(data[k,...].copy())
            data = dlist

    if not returnProj:
        return data

    else:
        return data, profile


def nodataToNan(inarr, listofvals):
    """
    Setting values to nan as needed
    """
    inarr = inarr.astype(float) # nans cannot be integers (i.e. in DEM)
    for val in listofvals:
        if val is not None:
            inarr[inarr == val] = np.nan


def rio_stats(fname, band=1, userNDV=None):
    if os.path.basename(fname).startswith('S1-GUNW'):
        fname = os.path.join(f'NETCDF:"{fname}":science/grids/data/unwrappedPhase')

    if os.path.exists(fname + '.vrt'):
        fname = fname + '.vrt'

    # Turn off PAM to avoid creating .aux.xml files
    with rasterio.Env(GDAL_PAM_ENABLED="NO"):
        with rasterio.open(fname) as src:
            gt    = src.transform.to_gdal()
            proj  = src.crs
            stats = src.statistics(band)

    return stats, proj, gt


def get_file_and_band(filestr):
    """
    Support file;bandnum as input for filename strings
    """
    parts = filestr.split(";")

    # Defaults to first band if no bandnum is provided
    if len(parts) == 1:
        return filestr.strip(), 1
    elif len(parts) == 2:
        return parts[0].strip(), int(parts[1].strip())
    else:
        raise ValueError(
            f"Cannot interpret {filestr} as valid filename"
        )


def writeResultsToXarray(dt, xpts, ypts, zpts, crs, wetDelay, hydroDelay, weather_model_file, out_type):
    '''
    write a 1-D array to a NETCDF5 file
    '''
       # Modify this as needed for NISAR / other projects
    ds = xarray.Dataset(
        data_vars=dict(
            wet=(["z", "y", "x"],
                 wetDelay,
                 {"units" : "m",
                  "description": f"wet {out_type} delay",
                  # 'crs': crs.to_epsg(),
                  "grid_mapping": "crs",

                 }),
            hydro=(["z", "y", "x"],
                   hydroDelay,
                   {"units": "m",
                    # 'crs': crs.to_epsg(),
                    "description": f"hydrostatic {out_type} delay",
                    "grid_mapping": "crs",
                   }),
        ),
        coords=dict(
            x=(["x"], xpts),
            y=(["y"], ypts),
            z=(["z"], zpts),
        ),
        attrs=dict(
            Conventions="CF-1.7",
            title="RAiDER geo cube",
            source=os.path.basename(weather_model_file),
            history=str(datetime.utcnow()) + " RAiDER",
            description=f"RAiDER geo cube - {out_type}",
            reference_time=str(dt),
        ),
    )

    # Write projection system mapping
    ds["crs"] = int(-2147483647) # dummy placeholder
    for k, v in crs.to_cf().items():
        ds.crs.attrs[k] = v

    # Write z-axis information
    ds.z.attrs["axis"] = "Z"
    ds.z.attrs["units"] = "m"
    ds.z.attrs["description"] = "height above ellipsoid"

    # If in degrees
    if crs.axis_info[0].unit_name == "degree":
        ds.y.attrs["units"] = "degrees_north"
        ds.y.attrs["standard_name"] = "latitude"
        ds.y.attrs["long_name"] = "latitude"

        ds.x.attrs["units"] = "degrees_east"
        ds.x.attrs["standard_name"] = "longitude"
        ds.x.attrs["long_name"] = "longitude"

    else:
        ds.y.attrs["axis"] = "Y"
        ds.y.attrs["standard_name"] = "projection_y_coordinate"
        ds.y.attrs["long_name"] = "y-coordinate in projected coordinate system"
        ds.y.attrs["units"] = "m"

        ds.x.attrs["axis"] = "X"
        ds.x.attrs["standard_name"] = "projection_x_coordinate"
        ds.x.attrs["long_name"] = "x-coordinate in projected coordinate system"
        ds.x.attrs["units"] = "m"

    return ds


def writeArrayToRaster(array, filename, noDataValue=0., fmt='ENVI', proj=None, gt=None):
    '''
    write a numpy array to a GDAL-readable raster
    '''
    array_shp = np.shape(array)
    if array.ndim != 2:
        raise RuntimeError('writeArrayToRaster: cannot write an array of shape {} to a raster image'.format(array_shp))

    # Data type
    if "complex" in str(array.dtype):
        dtype = np.complex64
    elif "float" in str(array.dtype):
        dtype = np.float32
    else:
        dtype = np.uint8

    # Geotransform
    trans = None
    if gt is not None:
        trans = rasterio.Affine.from_gdal(*gt)

    ## cant write netcdfs with rasterio in a simple way
    if fmt == 'nc':
        fmt = 'GTiff'
        filename = filename.replace('.nc', '.GTiff')

    with rasterio.open(filename, mode="w", count=1,
                       width=array_shp[1], height=array_shp[0],
                       dtype=dtype, crs=proj, nodata=noDataValue,
                       driver=fmt, transform=trans) as dst:
        dst.write(array, 1)
    logger.info('Wrote: %s', filename)
    return


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
    return G1 * (1 - 0.002637 * cosd(2 * lats) + 0.0000059 * (cosd(2 * lats))**2)


def get_Re(lats):
    '''
    Returns earth radius as a function of latitude for WGS84

    Args:
        lats    - ndarray of geodetic latitudes in degrees

    Returns:
        ndarray of earth radius at each latitude

    Example:
    >>> import numpy as np
    >>> from RAiDER.utilFcns import get_Re
    >>> output = get_Re(np.array([0, 30, 45, 60, 90]))
    >>> output
     array([6378137., 6372770.5219805, 6367417.56705189, 6362078.07851428, 6356752.])
    >>> assert output[0] == 6378137 # (Rmax)
    >>> assert output[-1] == 6356752 # (Rmin)
    '''
    return np.sqrt(1 / (((cosd(lats)**2) / Rmax**2) + ((sind(lats)**2) / Rmin**2)))


def geo_to_ht(lats, hts):
    """
    Convert geopotential height to ellipsoidal heights referenced to WGS84.

    Note that this formula technically computes height above geoid (geometric height)
    but the geoid is actually a perfect sphere;
    Thus returned heights are above a reference ellipsoid, which most assume to be
    a sphere (e.g., ECMWF - see https://confluence.ecmwf.int/display/CKB/ERA5%3A+compute+pressure+and+geopotential+on+model+levels%2C+geopotential+height+and+geometric+height#ERA5:computepressureandgeopotentialonmodellevels,geopotentialheightandgeometricheight-Geopotentialheight
    - "Geometric Height" and also https://confluence.ecmwf.int/display/CKB/ERA5%3A+data+documentation#ERA5:datadocumentation-Earthmodel).
    However, by calculating the ellipsoid here we directly reference to WGS84.

    Compare to MetPy:
    (https://unidata.github.io/MetPy/latest/api/generated/metpy.calc.geopotential_to_height.html)
    # h = (geopotential * Re) / (g0 * Re - geopotential)
    # Assumes a sphere instead of an ellipsoid

    Args:
        lats    - latitude of points of interest
        hts     - geopotential height at points of interest

    Returns:
        ndarray: geometric heights. These are approximate ellipsoidal heights referenced to WGS84
    """
    g_ll = _get_g_ll(lats) # gravity function of latitude
    Re = get_Re(lats) # Earth radius function of latitude

    # Calculate Geometric Height, h
    h = (hts * Re) / (g_ll / g0 * Re - hts)

    return h


def padLower(invar):
    '''
    add a layer of data below the lowest current z-level at height zmin
    '''
    new_var = _least_nonzero(invar)
    return np.concatenate((new_var[:, :, np.newaxis], invar), axis=2)


def checkShapes(los, lats, lons, hts):
    '''
    Make sure that by the time the code reaches here, we have a
    consistent set of line-of-sight and position data.
    '''
    from RAiDER.losreader import Zenith
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


def read_hgt_file(filename):
    '''
    Read height data from a comma-delimited file
    '''
    data = pd.read_csv(filename)
    hgts = data['Hgt_m'].values
    return hgts


def round_time(dt, roundTo=60):
    '''
    Round a datetime object to any time lapse in seconds
    dt: datetime.datetime object
    roundTo: Closest number of seconds to round to, default 1 minute.
    Source: https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object/10854034#10854034
    '''
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def writeDelays(aoi, wetDelay, hydroDelay,
                wetFilename, hydroFilename=None,
                outformat=None, ndv=0.):
    """ Write the delay numpy arrays to files in the format specified """

    # Need to consistently handle noDataValues
    wetDelay[np.isnan(wetDelay)] = ndv
    hydroDelay[np.isnan(hydroDelay)] = ndv

    # Do different things, depending on the type of input
    if aoi.type() == 'station_file':
        #TODO: why is this a try/except?
        try:
            df = pd.read_csv(aoi._filename).drop_duplicates(subset=["Lat", "Lon"])
        except ValueError:
            df = pd.read_csv(aoi._filename).drop_duplicates(subset=["Lat", "Lon"])

        df['wetDelay'] = wetDelay
        df['hydroDelay'] = hydroDelay
        df['totalDelay'] = wetDelay + hydroDelay
        df.to_csv(wetFilename, index=False)
        logger.info('Wrote delays to: %s', wetFilename)

    else:
        proj = aoi.projection()
        gt   = aoi.geotransform()
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


def unproject(z, l, x, y):
    if z not in _projections:
        _projections[z] = pyproj.Proj(proj='utm', zone=z, ellps='WGS84')
    if l < 'N':
        y -= 10000000
    lng, lat = _projections[z](x, y, inverse=True)
    return (lng, lat)


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
    return np.reshape(lon, shp), np.reshape(lat, shp)


def transform_bbox(snwe_in, dest_crs=4326, src_crs=4326, margin=100.):
    """
    Transform bbox to lat/lon or another CRS for use with rest of workflow
    Returns: SNWE
    """
    # TODO - Handle dateline crossing
    if isinstance(src_crs, int):
        src_crs = pyproj.CRS.from_epsg(src_crs)
    elif isinstance(src_crs, str):
        src_crs = pyproj.CRS(src_crs)

    # Handle margin for input bbox in degrees
    if src_crs.axis_info[0].unit_name == "degree":
        margin = margin / 1.0e5

    if isinstance(dest_crs, int):
        dest_crs = pyproj.CRS.from_epsg(dest_crs)
    elif isinstance(dest_crs, str):
        dest_crs = pyproj.CRS(dest_crs)

    # If dest_crs is same as src_crs
    if dest_crs == src_crs:
        return snwe_in

    T = Transformer.from_crs(src_crs, dest_crs, always_xy=True)
    xs = np.linspace(snwe_in[2]-margin, snwe_in[3]+margin, num=11)
    ys = np.linspace(snwe_in[0]-margin, snwe_in[1]+margin, num=11)
    X, Y = np.meshgrid(xs, ys)

    # Transform to lat/lon
    xx, yy = T.transform(X, Y)

    # query_area convention
    snwe = [np.nanmin(yy), np.nanmax(yy),
            np.nanmin(xx), np.nanmax(xx)]
    return snwe


def clip_bbox(bbox, spacing):
    """
    Clip box to multiple of spacing
    """
    return [np.floor(bbox[0] / spacing) * spacing,
            np.ceil(bbox[1] / spacing) * spacing,
            np.floor(bbox[2] / spacing) * spacing,
            np.ceil(bbox[3] / spacing) * spacing]


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


def writeWeatherVars2NETCDF4(self, lat, lon, h, q, p, t, outName=None, NoDataValue=None, chunk=(1, 91, 144), mapping_name='WGS84'):
    '''
    By calling the abstract/modular netcdf writer (RAiDER.utilFcns.write2NETCDF4core), write the OpenDAP/PyDAP-retrieved weather model data (GMAO and MERRA-2) to a NETCDF4 file
    that can be accessed by external programs.

    The point of doing this is to alleviate some of the memory load of keeping
    the full model in memory and make it easier to scale up the program.
    '''

    import netCDF4

    if outName is None:
        outName = os.path.join(
            os.getcwd() + '/weather_files',
            self._Name + datetime.strftime(
                self._time, '_%Y_%m_%d_T%H_%M_%S'
            ) + '.nc'
        )

    if NoDataValue is None:
        NoDataValue = -9999.

    self._time = getTimeFromFile(outName)

    dimidZ, dimidY, dimidX = t.shape
    chunk_lines_Y = np.min([chunk[1], dimidY])
    chunk_lines_X = np.min([chunk[2], dimidX])
    ChunkSize = [1, chunk_lines_Y, chunk_lines_X]

    nc_outfile = netCDF4.Dataset(outName, 'w', clobber=True, format='NETCDF4')
    nc_outfile.setncattr('Conventions', 'CF-1.6')
    nc_outfile.setncattr('datetime', datetime.strftime(self._time, "%Y_%m_%dT%H_%M_%S"))
    nc_outfile.setncattr('date_created', datetime.now().strftime("%Y_%m_%dT%H_%M_%S"))
    title = self._Name + ' weather model data'
    nc_outfile.setncattr('title', title)

    tran = [lon[0], lon[1] - lon[0], 0.0, lat[0], 0.0, lat[1] - lat[0]]

    dimension_dict = {
        'x': {'varname': 'x',
              'datatype': np.dtype('float64'),
              'dimensions': ('x'),
              'length': dimidX,
              'FillValue': None,
              'standard_name': 'longitude',
              'description': 'longitude',
              'dataset': lon,
              'units': 'degrees_east'},
        'y': {'varname': 'y',
              'datatype': np.dtype('float64'),
              'dimensions': ('y'),
              'length': dimidY,
              'FillValue': None,
              'standard_name': 'latitude',
              'description': 'latitude',
              'dataset': lat,
              'units': 'degrees_north'},
        'z': {'varname': 'z',
              'datatype': np.dtype('float32'),
              'dimensions': ('z'),
              'length': dimidZ,
              'FillValue': None,
              'standard_name': 'model_layers',
              'description': 'model layers',
              'dataset': np.arange(dimidZ),
              'units': 'layer'}
    }

    dataset_dict = {
        'h': {'varname': 'H',
              'datatype': np.dtype('float32'),
              'dimensions': ('z', 'y', 'x'),
              'grid_mapping': mapping_name,
              'FillValue': NoDataValue,
              'ChunkSize': ChunkSize,
              'standard_name': 'mid_layer_heights',
              'description': 'mid layer heights',
              'dataset': h,
              'units': 'm'},
        'q': {'varname': 'QV',
              'datatype': np.dtype('float32'),
              'dimensions': ('z', 'y', 'x'),
              'grid_mapping': mapping_name,
              'FillValue': NoDataValue,
              'ChunkSize': ChunkSize,
              'standard_name': 'specific_humidity',
              'description': 'specific humidity',
              'dataset': q,
              'units': 'kg kg-1'},
        'p': {'varname': 'PL',
              'datatype': np.dtype('float32'),
              'dimensions': ('z', 'y', 'x'),
              'grid_mapping': mapping_name,
              'FillValue': NoDataValue,
              'ChunkSize': ChunkSize,
              'standard_name': 'mid_level_pressure',
              'description': 'mid level pressure',
              'dataset': p,
              'units': 'Pa'},
        't': {'varname': 'T',
              'datatype': np.dtype('float32'),
              'dimensions': ('z', 'y', 'x'),
              'grid_mapping': mapping_name,
              'FillValue': NoDataValue,
              'ChunkSize': ChunkSize,
              'standard_name': 'air_temperature',
              'description': 'air temperature',
              'dataset': t,
              'units': 'K'}
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
        crs = pyproj.CRS.from_epsg(epsg)

        grid_mapping = 'WGS84'  # need to set this as an attribute for the image variables
    else:
        crs = pyproj.CRS.from_wkt(mapping_name)
        grid_mapping = 'CRS'

    datatype = np.dtype('S1')
    dimensions = ()
    var = nc_outfile.createVariable(
        grid_mapping,
        datatype,
        dimensions,
        fill_value=None
    )

    # variable made, now add attributes
    for k, v in crs.to_cf().items():
        var.setncattr(k, v)

    var.setncattr('GeoTransform', ' '.join(str(x) for x in tran))  # note this has pixel size in it - set  explicitly above

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
        # Override with correct name here
        var.setncattr('grid_mapping', grid_mapping) # dataset_dict[data]['grid_mapping'])
        var.setncattr('standard_name', dataset_dict[data]['standard_name'])
        var.setncattr('description', dataset_dict[data]['description'])
        if 'units' in dataset_dict[data]:
            var.setncattr('units', dataset_dict[data]['units'])

        ndmask = np.isnan(dataset_dict[data]['dataset'])
        dataset_dict[data]['dataset'][ndmask] = FillValue

        var[:] = dataset_dict[data]['dataset'].astype(datatype)

    return nc_outfile


def convertLons(inLons):
    '''Convert lons from 0-360 to -180-180'''
    mask = inLons > 180
    outLons = inLons
    outLons[mask] = outLons[mask] - 360
    return outLons


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


def read_EarthData_loginInfo(filepath=None):

    from netrc import netrc

    urs_usr, _, urs_pwd = netrc().hosts["urs.earthdata.nasa.gov"]
    return urs_usr, urs_pwd


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


def calcgeoh(lnsp, t, q, z, a, b, R_d, num_levels):
    '''
    Calculate pressure, geopotential, and geopotential height
    from the surface pressure and model levels provided by a weather model.
    The model levels are numbered from the highest eleveation to the lowest.
    Args:
    ----------
        lnsp: ndarray         - [y, x] array of log surface pressure
        t: ndarray            - [z, y, x] cube of temperatures
        q: ndarray            - [z, y, x] cube of specific humidity
        geopotential: ndarray - [z, y, x] cube of geopotential values
        a: ndarray            - [z] vector of a values
        b: ndarray            - [z] vector of b values
        num_levels: int       - integer number of model levels
    Returns:
    -------
        geopotential - The geopotential in units of height times acceleration
        pressurelvs  - The pressure at each of the model levels for each of
                       the input points
        geoheight    - The geopotential heights
    '''
    geopotential = np.zeros_like(t)
    pressurelvs = np.zeros_like(geopotential)
    geoheight = np.zeros_like(geopotential)

    # log surface pressure
    # Note that we integrate from the ground up, so from the largest model level to 0
    sp = np.exp(lnsp)

    if len(a) != num_levels + 1 or len(b) != num_levels + 1:
        raise ValueError(
            'I have here a model with {} levels, but parameters a '.format(num_levels) +
            'and b have lengths {} and {} respectively. Of '.format(len(a), len(b)) +
            'course, these three numbers should be equal.')

    # Integrate up into the atmosphere from *lowest level*
    z_h = 0  # initial value
    for lev, t_level, q_level in zip(
            range(num_levels, 0, -1), t[::-1], q[::-1]):

        # lev is the level number 1-60, we need a corresponding index
        # into ts and qs
        # ilevel = num_levels - lev # << this was Ray's original, but is a typo
        # because indexing like that results in pressure and height arrays that
        # are in the opposite orientation to the t/q arrays.
        ilevel = lev - 1

        # compute moist temperature
        t_level = t_level * (1 + 0.609133 * q_level)

        # compute the pressures (on half-levels)
        Ph_lev = a[lev - 1] + (b[lev - 1] * sp)
        Ph_levplusone = a[lev] + (b[lev] * sp)

        pressurelvs[ilevel] = Ph_lev  # + Ph_levplusone) / 2  # average pressure at half-levels above and below

        if lev == 1:
            dlogP = np.log(Ph_levplusone / 0.1)
            alpha = np.log(2)
        else:
            dlogP = np.log(Ph_levplusone) - np.log(Ph_lev)
            alpha = 1 - ((Ph_lev / (Ph_levplusone - Ph_lev)) * dlogP)

        TRd = t_level * R_d

        # z_f is the geopotential of this full level
        # integrate from previous (lower) half-level z_h to the full level
        z_f = z_h + TRd * alpha + z

        # Geopotential (add in surface geopotential)
        geopotential[ilevel] = z_f
        geoheight[ilevel] = geopotential[ilevel] / g0

        # z_h is the geopotential of 'half-levels'
        # integrate z_h to next half level
        z_h += TRd * dlogP

    return geopotential, pressurelvs, geoheight


def transform_coords(proj1, proj2, x, y):
    """
    Transform coordinates from proj1 to proj2 (can be EPSG or crs from proj).
    e.g. x, y = transform_coords(4326, 4087, lon, lat)
    """
    transformer = Transformer.from_crs(proj1, proj2, always_xy=True)
    return transformer.transform(x, y)


def get_nearest_wmtimes(t0, time_delta):
    """"
    Get the nearest two available times to the requested time given a time step

    Args:
        t0         - user-requested Python datetime
        time_delta  - time interval of weather model

    Returns:
        tuple: list of datetimes representing the one or two closest
        available times to the requested time

    Example:
    >>> import datetime
    >>> from RAiDER.utilFcns import get_nearest_wmtimes
    >>> t0 = datetime.datetime(2020,1,1,11,35,0)
    >>> get_nearest_wmtimes(t0, 3)
     (datetime.datetime(2020, 1, 1, 9, 0), datetime.datetime(2020, 1, 1, 12, 0))
    """
    # get the closest time available
    tclose = round_time(t0, roundTo = time_delta * 60 *60)

    # Just calculate both options and take the closest
    t2_1 = tclose + timedelta(hours=time_delta)
    t2_2 = tclose - timedelta(hours=time_delta)
    t2 = [t2_1 if get_dt(t2_1, t0) < get_dt(t2_2, t0) else t2_2][0]

    # If you're within 5 minutes just take the closest time
    if get_dt(tclose, t0) < _THRESHOLD_SECONDS:
        return [tclose]
    else:
        if t2 > tclose:
            return [tclose, t2]
        else:
            return [t2, tclose]


def get_dt(t1,t2):
    '''
    Helper function for getting the absolute difference in seconds between
    two python datetimes

    Args:
        t1, t2  - Python datetimes

    Returns:
        Absolute difference in seconds between the two inputs

    Examples:
    >>> import datetime
    >>> from RAiDER.utilFcns import get_dt
    >>> get_dt(datetime.datetime(2020,1,1,5,0,0), datetime.datetime(2020,1,1,0,0,0))
     18000.0
    '''
    return np.abs((t1 - t2).total_seconds())
