"""Geodesy-related utility functions."""

import datetime as dt
import pathlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import rasterio
import xarray as xr
import yaml
from numpy import ndarray
from pyproj import CRS, Proj, Transformer

import RAiDER
from RAiDER.constants import (
    R_EARTH_MAX_WGS84 as Rmax,
)
from RAiDER.constants import (
    R_EARTH_MIN_WGS84 as Rmin,
)
from RAiDER.constants import (
    _THRESHOLD_SECONDS,
)
from RAiDER.constants import (
    _g0 as g0,
)
from RAiDER.constants import (
    _g1 as G1,
)
from RAiDER.llreader import AOI
from RAiDER.logger import logger
from RAiDER.types import BB, RIO, CRSLike


# Optional imports
try:
    import pandas as pd
except ImportError:
    pd = None
try:
    import multiprocessing as mp
except ImportError:
    mp = None
try:
    import progressbar
except ImportError:
    progressbar = None


pbar = None


def projectDelays(delay: Union[float, np.array], inc: Union[float, np.array]) -> Union[float, np.array]:
    """Project zenith delays to LOS."""
    if inc == 90:
        raise ZeroDivisionError
    return delay / cosd(inc)


def floorish(val: Union[float, np.array], frac: Union[float, np.array]) -> Union[float, np.array]:
    """Round a value to the lower fractional part."""
    return val - (val % frac)


def sind(x: Union[float, np.array]) -> Union[float, np.array]:
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def cosd(x: Union[float, np.array]) -> Union[float, np.array]:
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def lla2ecef(lat: Union[float, np.array], lon: Union[float, np.array], height: Union[float, np.array]) -> np.array:
    """Transforms from lla to ecef."""
    T = Transformer.from_crs(4326, 4978, always_xy=True)

    return T.transform(lon, lat, height)


def ecef2lla(x: Union[float, np.array], y: Union[float, np.array], z: Union[float, np.array]) -> np.array:
    """Converts ecef to lla."""
    T = Transformer.from_crs(4978, 4326, always_xy=True)

    return T.transform(x, y, z)


def enu2ecef(
    east: ndarray,
    north: ndarray,
    up: ndarray,
    lat0: ndarray,
    lon0: ndarray,
    h0: ndarray,
) -> np.array:
    """Converts enu to ecef."""
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


def ecef2enu(xyz: Union[float, np.array], 
             lat: Union[float, np.array], 
             lon: Union[float, np.array], 
             height: Union[float, np.array]) -> np.array:
    """Convert ECEF xyz to ENU."""
    """height is not used here, needs looked at"""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]

    t = cosd(lon) * x + sind(lon) * y

    e = -sind(lon) * x + cosd(lon) * y
    n = -sind(lat) * t + cosd(lat) * z
    u = cosd(lat) * t + sind(lat) * z
    return np.stack((e, n, u), axis=-1)


def rio_profile(path: Path) -> RIO.Profile:
    """Reads the profile of a rasterio file."""
    path_vrt = Path(f'{path}.vrt')

    if path.name.startswith('S1-GUNW'):
        # need to access subdataset directly
        path = Path(f'NETCDF:"{path}":science/grids/data/unwrappedPhase')
    elif path_vrt.exists():
        path = path_vrt

    with rasterio.open(path) as src:
        return src.profile


def rio_extents(profile: RIO.Profile) -> BB.SNWE:
    """Get a bounding box in SNWE from a rasterio profile."""
    gt = profile['transform'].to_gdal()
    xSize = profile['width']
    ySize = profile['height']
    W, E = gt[0], gt[0] + (xSize - 1) * gt[1] + (ySize - 1) * gt[2]
    N, S = gt[3], gt[3] + (xSize - 1) * gt[4] + (ySize - 1) * gt[5]
    return S, N, W, E


def rio_open(
    path: Union[Path, str],
    userNDV: Optional[float]=None,
    band: Optional[int]=None
) -> tuple[np.ndarray, RIO.Profile]:
    """Reads a rasterio-compatible raster file and returns the data and profile."""
    path = Path(path)
    vrt_path = path.with_suffix(path.suffix + '.vrt')
    if vrt_path.exists():
        path = vrt_path

    with rasterio.open(path) as src:
        profile: RIO.Profile = src.profile

        # For all bands
        nodata: tuple[float, ...] = src.nodatavals

        # If user requests a band
        if band is not None:
            ndv = nodata[band - 1]
            data: np.ndarray = src.read(band).squeeze()
            nodataToNan(data, [userNDV, ndv])

        else:
            data: np.ndarray = src.read().squeeze()
            if data.ndim > 2:
                for bnd in range(data.shape[0]):
                    val = data[bnd, ...]
                    nodataToNan(val, [userNDV, nodata[bnd]])
            else:
                nodataToNan(data, list(nodata) + [userNDV])

        if data.ndim > 2:
            dlist: list[list[float]] = []
            for k in range(data.shape[0]):
                dlist.append(data[k].copy())
            data = np.array(dlist)

    return data, profile


def nodataToNan(inarr: np.ndarray, vals: list[Optional[float]]) -> None:
    """Setting values to nan as needed."""
    inarr = inarr.astype(float)  # nans cannot be integers (i.e. in DEM)
    for val in vals:
        if val is not None:
            inarr[inarr == val] = np.nan


def rio_stats(path: Path, band: int=1) -> tuple[RIO.Statistics, Optional[CRS], RIO.GDAL]:
    """Read a rasterio-compatible file and pull the metadata.

    Args:
        path: Path
            file path to be loaded
        band: int
            band number to use for getting statistics

    Returns:
        stats   - a list of stats for the specified band
        proj    - CRS/projection information for the file
        gt      - geotransform for the data
    """
    if path.name.startswith('S1-GUNW'):
        path = Path(f'NETCDF:"{path}":science/grids/data/unwrappedPhase')

    vrt_path = path.with_suffix(path.suffix + '.vrt')
    if vrt_path.exists():
        path = vrt_path

    # Turn off PAM to avoid creating .aux.xml files
    with rasterio.Env(GDAL_PAM_ENABLED='NO'):
        with rasterio.open(path) as src:
            stats = src.statistics(band)
            proj = src.crs
            gt = src.transform.to_gdal()

    return stats, proj, gt


def get_file_and_band(filestr: str) -> tuple[Path, int]:
    """Support file;bandnum as input for filename strings."""
    parts = filestr.split(';')

    # Defaults to first band if no bandnum is provided
    if len(parts) == 1:
        return Path(filestr.strip()), 1
    elif len(parts) == 2:
        return Path(parts[0].strip()), int(parts[1].strip())
    else:
        raise ValueError(f'Cannot interpret {filestr} as valid filename')


def writeArrayToRaster(
    array: np.ndarray,
    path: Path,
    noDataValue: float=0.0,
    fmt: str='ENVI',
    proj: Optional[CRS]=None,
    gt: Optional[RIO.GDAL]=None
) -> None:
    """Write a numpy array to a GDAL-readable raster."""
    array_shp = np.shape(array)
    if array.ndim != 2:
        raise RuntimeError(f'writeArrayToRaster: cannot write an array of shape {array_shp} to a raster image')

    # Data type
    if 'complex' in str(array.dtype):
        dtype = np.complex64
    elif 'float' in str(array.dtype):
        dtype = np.float32
    else:
        dtype = np.uint8

    # Geotransform
    trans = None
    if gt is not None:
        try:
            trans = rasterio.Affine.from_gdal(*gt)
        except TypeError:
            trans = gt

    # cant write netcdfs with rasterio in a simple way
    if fmt == 'nc':
        fmt = 'GTiff'
        path = path.with_suffix('.tif')

    with rasterio.open(
        path,
        mode='w',
        count=1,
        width=array_shp[1],
        height=array_shp[0],
        dtype=dtype,
        crs=proj,
        nodata=noDataValue,
        driver=fmt,
        transform=trans,
    ) as dst:
        dst.write(array, 1)
    logger.info('Wrote: %s', path)


def round_date(date: dt.datetime, precision: int) -> dt.datetime:
    """Rounds the date to the nearest precision in seconds."""
    # First try rounding up
    # Timedelta since the beginning of time
    T0 = dt.datetime.min

    try:
        datedelta = T0 - date
    except TypeError:
        T0 = T0.replace(tzinfo=dt.timezone(offset=dt.timedelta()))
        datedelta = T0 - date

    # Round that dt.timedelta to the specified precision
    rem = datedelta % precision
    # Add back to get date rounded up
    round_up = date + rem

    # Next try rounding down
    try:
        datedelta = date - T0
    except TypeError:
        T0 = T0.replace(tzinfo=dt.timezone(offset=dt.timedelta()))
        datedelta = date - T0

    rem = datedelta % precision
    round_down = date - rem

    # It's not the most efficient to calculate both and then choose, but
    # it's clear, and performance isn't critical here.
    up_diff = round_up - date
    down_diff = date - round_down

    return round_up if up_diff < down_diff else round_down


def _least_nonzero(a: np.array) -> np.array:
    """Fill in a flat array with the first non-nan value in the last dimension.

    Useful for interpolation below the bottom of the weather model.
    """
    mgrid_index = tuple(slice(None, d) for d in a.shape[:-1])
    return a[tuple(np.mgrid[mgrid_index]) + ((~np.isnan(a)).argmax(-1),)]


def _get_g_ll(lats: Union[float, np.array]) -> Union[float, np.array]:
    """Compute the variation in gravity constant with latitude."""
    return G1 * (1 - 0.002637 * cosd(2 * lats) + 0.0000059 * (cosd(2 * lats)) ** 2)


def get_Re(lats: ndarray) -> ndarray:
    """
    Returns earth radius as a function of latitude for WGS84.

    Args: lats
        ndarray of geodetic latitudes in degrees

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
    """
    return np.sqrt(1 / (((cosd(lats) ** 2) / Rmax**2) + ((sind(lats) ** 2) / Rmin**2)))


def geo_to_ht(lats: ndarray, hts: ndarray) -> ndarray:
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
        lats: ndarray
            latitude of points of interest
        hts: ndarray
            geopotential height at points of interest
    

    Returns:
        ndarray: geometric heights. These are approximate ellipsoidal heights referenced to WGS84
    """
    g_ll = _get_g_ll(lats)  # gravity function of latitude
    Re = get_Re(lats)  # Earth radius function of latitude

    # Calculate Geometric Height, h
    h = (hts * Re) / (g_ll / g0 * Re - hts)

    return h


def padLower(invar: np.array) -> np.array:
    """Add a layer of data below the lowest current z-level at height zmin."""
    new_var = _least_nonzero(invar)
    return np.concatenate((new_var[:, :, np.newaxis], invar), axis=2)


def round_time(datetime: dt.datetime, roundTo: int=60) -> dt.datetime:
    """Round a datetime object to any time lapse in seconds."""
    """
    datetime: dt.datetime object
    roundTo: Closest number of seconds to round to, default 1 minute.
    Source: https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object/10854034#10854034
    """
    seconds = (datetime.replace(tzinfo=None) - datetime.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return datetime + dt.timedelta(0, rounding - seconds, -datetime.microsecond)


def writeDelays(
    aoi: AOI,  #: AOI,
    wetDelay: ndarray,
    hydroDelay: ndarray,
    wet_path: Path,
    hydro_path: Optional[Path]=None,
    outformat: str=None,
    ndv: float=0.0
) -> None:
    """Write the delay numpy arrays to files in the format specified."""
    if pd is None:
        raise ImportError('pandas is required to write GNSS delays to a file')

    # Need to consistently handle noDataValues
    wetDelay[np.isnan(wetDelay)] = ndv
    hydroDelay[np.isnan(hydroDelay)] = ndv

    # Do different things, depending on the type of input
    if aoi.type() == 'station_file':
        df = pd.read_csv(aoi._filename).drop_duplicates(subset=['Lat', 'Lon'])

        df['wetDelay'] = wetDelay
        df['hydroDelay'] = hydroDelay
        df['totalDelay'] = wetDelay + hydroDelay
        df.to_csv(str(wet_path), index=False)
        logger.info('Wrote delays to: %s', wet_path.absolute())

    else:
        if hydro_path is None:
            raise ValueError('Hydro delay file path must be specified if the AOI is not a station file')
        proj = aoi.projection()
        gt = aoi.geotransform()
        writeArrayToRaster(wetDelay, wet_path, noDataValue=ndv, fmt=outformat, proj=proj, gt=gt)
        writeArrayToRaster(hydroDelay, hydro_path, noDataValue=ndv, fmt=outformat, proj=proj, gt=gt)


def getTimeFromFile(filename: Union[str, Path]) -> dt.datetime:
    """Parse a filename to get a date-time."""
    fmt = '%Y_%m_%d_T%H_%M_%S'
    p = re.compile(r'\d{4}_\d{2}_\d{2}_T\d{2}_\d{2}_\d{2}')
    out = p.search(filename).group()
    return dt.datetime.strptime(out, fmt)


# Part of the following UTM and WGS84 converter is borrowed from https://gist.github.com/twpayne/4409500
# Credits go to Tom Payne

_projections = {}


def zone(coordinates: Union[list, tuple, np.array]) -> int:
    """Returns the zone of a UTM zone."""
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


def letter(coordinates: Union[list, tuple, np.array]) -> str:
    """Returns zone UTM letter."""
    return 'CDEFGHJKLMNPQRSTUVWXX'[int((coordinates[1] + 80) / 8)]


def project(coordinates: Union[list, tuple, np.array], z: int=None, ltr: str=None) -> tuple[int, str, float, float]:
    """Returns zone UTM coordinate."""
    if z is None:
        z = zone(coordinates)
    if ltr is None:
        ltr = letter(coordinates)
    if z not in _projections:
        _projections[z] = Proj(proj='utm', zone=z, ellps='WGS84')
    x, y = _projections[z](coordinates[0], coordinates[1])
    if y < 0:
        y += 10000000
    return z, ltr, x, y


def unproject(z: int, ltr: str, x: float, y: float) -> tuple[Union[float, np.array]]:
    """Returns a tuple containing the zone UTM lng and lat."""
    if z not in _projections:
        _projections[z] = Proj(proj='utm', zone=z, ellps='WGS84')
    if ltr < 'N':
        y -= 10000000
    lng, lat = _projections[z](x, y, inverse=True)
    return (lng, lat)


def WGS84_to_UTM(lon: float, lat: float, common_center: bool=False) -> tuple[np.array]:
    """Converts WGS84 to UTM."""
    shp = lat.shape
    lon = np.ravel(lon)
    lat = np.ravel(lat)
    if common_center:
        lon0 = np.median(lon)
        lat0 = np.median(lat)
        z0, l0, _, _ = project((lon0, lat0))
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


def UTM_to_WGS84(z: np.array, ltr: np.array, x: np.array, y: np.array) -> tuple[np.array]:
    """Converts UTM to WGS84."""
    # Ensure inputs are numpy arrays
    z = np.ravel(z)
    ltr = np.ravel(ltr)
    x = np.ravel(x)
    y = np.ravel(y)
    
    # Validate that all input arrays have the same length
    if not (len(z) == len(ltr) == len(x) == len(y)):
        raise ValueError("All input arrays must have the same length.")
    
    # Initialize arrays for lat and lon
    lat = np.empty_like(x, dtype=float)
    lon = np.empty_like(x, dtype=float)
    
    # Iterate over all coordinates
    for ind in range(len(z)):
        zz = z[ind]
        ll = ltr[ind]
        xx = x[ind]
        yy = y[ind]
        
        # Perform the transformation
        coordinates = unproject(zz, ll, xx, yy)
        
        # Assign the results to lat and lon
        lon[ind] = coordinates[0]
        lat[ind] = coordinates[1]
    
    # Reshape and return the results
    return np.reshape(lon, x.shape), np.reshape(lat, x.shape)


def transform_bbox(snwe_in: list, dest_crs: int=4326, src_crs: int=4326, buffer: float=100.0) -> Tuple[np.array]:
    """Transform bbox to lat/lon or another CRS for use with rest of workflow."""
    """
    Returns: SNWE
    """
    # TODO - Handle dateline crossing
    if isinstance(src_crs, int):
        src_crs = CRS.from_epsg(src_crs)
    elif isinstance(src_crs, str):
        src_crs = CRS(src_crs)

    # Handle margin for input bbox in degrees
    if src_crs.axis_info[0].unit_name == 'degree':
        buffer = buffer / 1.0e5

    if isinstance(dest_crs, int):
        dest_crs = CRS.from_epsg(dest_crs)
    elif isinstance(dest_crs, str):
        dest_crs = CRS(dest_crs)

    # If dest_crs is same as src_crs
    if dest_crs == src_crs:
        return snwe_in

    T = Transformer.from_crs(src_crs, dest_crs, always_xy=True)
    xs = np.linspace(snwe_in[2] - buffer, snwe_in[3] + buffer, num=11)
    ys = np.linspace(snwe_in[0] - buffer, snwe_in[1] + buffer, num=11)
    X, Y = np.meshgrid(xs, ys)

    # Transform to lat/lon
    xx, yy = T.transform(X, Y)

    # query_area convention
    snwe = [np.nanmin(yy), np.nanmax(yy), np.nanmin(xx), np.nanmax(xx)]
    return snwe


def clip_bbox(bbox: Union[list, tuple, ndarray], spacing: Union[int, float]) -> List[np.array]:
    """Clip box to multiple of spacing."""
    return [
        np.floor(bbox[0] / spacing) * spacing,
        np.ceil(bbox[1] / spacing) * spacing,
        np.floor(bbox[2] / spacing) * spacing,
        np.ceil(bbox[3] / spacing) * spacing,
    ]


def requests_retry_session(retries: int=10, session=None):  # noqa: ANN001, ANN201
    """https://www.peterbe.com/plog/best-practice-with-retries-with-requests."""
    import requests
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry

    # add a retry strategy; https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
    session = session or requests.Session()
    retry = Retry(
        total=retries, read=retries, connect=retries, backoff_factor=0.3, status_forcelist=list(range(429, 505))
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session


def writeWeatherVarsXarray(lat: float, lon: float, h: float, q: float, p: float, t: float, datetime: dt.datetime, crs: float, outName: str=None, NoDataValue: int=-9999, chunk: list=(1, 91, 144)) -> None:
    """Does not return anything."""
    # I added datetime as an input to the function and just copied these two lines from merra2 for the attrs_dict
    attrs_dict = {
        'datetime': datetime.strftime('%Y_%m_%dT%H_%M_%S'),
        'date_created': datetime.now().strftime('%Y_%m_%dT%H_%M_%S'),
        'NoDataValue': NoDataValue,
        'chunksize': chunk,
        # 'mapping_name': mapping_name,
    }

    dimension_dict = {
        'latitude': (('y', 'x'), lat),
        'longitude': (('y', 'x'), lon),
    }

    dataset_dict = {
        'h': (('z', 'y', 'x'), h),
        'q': (('z', 'y', 'x'), q),
        'p': (('z', 'y', 'x'), p),
        't': (('z', 'y', 'x'), t),
    }

    ds = xr.Dataset(
        data_vars=dataset_dict,
        coords=dimension_dict,
        attrs=attrs_dict,
    )

    ds['h'].attrs['standard_name'] = 'mid_layer_heights'
    ds['p'].attrs['standard_name'] = 'mid_level_pressure'
    ds['q'].attrs['standard_name'] = 'specific_humidity'
    ds['t'].attrs['standard_name'] = 'air_temperature'

    ds['h'].attrs['units'] = 'm'
    ds['p'].attrs['units'] = 'Pa'
    ds['q'].attrs['units'] = 'kg kg-1'
    ds['t'].attrs['units'] = 'K'

    ds['proj'] = 0
    for k, v in crs.to_cf().items():
        ds.proj.attrs[k] = v
    for var in ds.data_vars:
        ds[var].attrs['grid_mapping'] = 'proj'

    ds.to_netcdf(outName)
    del ds


def convertLons(inLons: np.ndarray) -> np.ndarray:
    """Convert lons from 0-360 to -180-180."""
    mask = inLons > 180
    outLons = inLons
    outLons[mask] = outLons[mask] - 360
    return outLons


def read_NCMR_loginInfo(filepath: str = None) -> Tuple[str, str, str]:
    """Returns login information."""
    from pathlib import Path

    if filepath is None:
        filepath = str(Path.home()) + '/.ncmrlogin'

    with Path(filepath).open() as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError("The login file must have at least three lines")

    def parse_line(line, expected_key):  # noqa: ANN001, ANN202
        parts = line.strip().split(': ')
        if len(parts) != 2 or parts[0] != expected_key:
            raise ValueError(f"Improperly formatted login file: Expected '{expected_key}: <value>'")
        return parts[1]

    url = parse_line(lines[0], "url")
    username = parse_line(lines[1], "username")
    password = parse_line(lines[2], "password")

    return url, username, password


def read_EarthData_loginInfo(filepath: str = None) -> Tuple[str, str]:
    """Returns username and password."""
    from netrc import netrc

    nrc = netrc(filepath) if filepath else netrc()
    try:
        urs_usr, _, urs_pwd = nrc.hosts['urs.earthdata.nasa.gov']
        if not urs_usr or not urs_pwd:
            raise ValueError("Invalid login information in netrc")
        return urs_usr, urs_pwd
    except KeyError:
        raise KeyError("No entry for urs.earthdata.nasa.gov in netrc")


def show_progress(block_num: Union[int, float], block_size: Union[int, float], total_size: Union[int, float]) -> None:
    """Show download progress."""
    global pbar
    try:
        pbar
    except NameError:
        pbar = None
    
    if pbar is None:
        try:
            pbar = progressbar.ProgressBar(maxval=total_size)
        except NameError:
            raise ImportError('RAiDER.utilFcns: show_progress - progressbar is not available')
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None


def getChunkSize(in_shape: ndarray) -> Tuple:
    """Create a reasonable chunk size."""
    if mp is None:
        raise ImportError('RAiDER.utilFcns: getChunkSize - multiprocessing is not available')
    minChunkSize = 100
    maxChunkSize = 1000
    cpu_count = mp.cpu_count()
    chunkSize = tuple(max(min(maxChunkSize, s // cpu_count), min(s, minChunkSize)) for s in in_shape)
    return chunkSize


def calcgeoh(lnsp: ndarray, t: ndarray, q: ndarray, z: ndarray, a: ndarray, b: ndarray, R_d: float, num_levels: int) -> Tuple[np.ndarray]:
    """
    Calculate pressure, geopotential, and geopotential height
    from the surface pressure and model levels provided by a weather model.
    The model levels are numbered from the highest eleveation to the lowest.

    Args:
    ----------
        lnsp: ndarray         - [y, x] array of log surface pressure
        t: ndarray            - [z, y, x] cube of temperatures
        q: ndarray            - [z, y, x] cube of specific humidity
        z: ndarray - [z, y, x] cube of geopotential values
        a: ndarray            - [z] vector of a values
        b: ndarray            - [z] vector of b values
        R_d: float            - R_d from weather model
        num_levels: int       - integer number of model levels

    Returns:
    -------
        geopotential - The geopotential in units of height times acceleration
        pressurelvs  - The pressure at each of the model levels for each of
                       the input points
        geoheight    - The geopotential heights
    """
    geopotential = np.zeros_like(t)
    pressurelvs = np.zeros_like(geopotential)
    geoheight = np.zeros_like(geopotential)

    # log surface pressure
    # Note that we integrate from the ground up, so from the largest model level to 0
    sp = np.exp(lnsp)

    if len(a) != num_levels + 1 or len(b) != num_levels + 1:
        raise ValueError(
            f'I have here a model with {num_levels} levels, but parameters a and b have lengths {len(a)} and {len(b)} '
            'respectively. Of course, these three numbers should be equal.'
        )

    # Integrate up into the atmosphere from *lowest level*
    z_h = 0  # initial value
    for lev, t_level, q_level in zip(range(num_levels, 0, -1), t[::-1], q[::-1]):
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


def transform_coords(proj1: CRS, proj2: CRS, x: float, y: float) -> np.ndarray:
    """
    Transform coordinates from proj1 to proj2 (can be EPSG or crs from proj).
    e.g. x, y = transform_coords(4326, 4087, lon, lat).
    """
    transformer = Transformer.from_crs(proj1, proj2, always_xy=True)
    return transformer.transform(x, y)


def get_nearest_wmtimes(t0: dt.datetime, time_delta: int) -> List[dt.datetime]:
    """Get the nearest two available times to the requested time given a time step.

    Args:
        t0          - user-requested Python datetime
        time_delta  - time interval of weather model

    Returns:
        tuple: list of datetimes representing the one or two closest
        available times to the requested time

    Example:
    >>> import datetime as dt
    >>> from RAiDER.utilFcns import get_nearest_wmtimes
    >>> t0 = dt.datetime(2020,1,1,11,35,0)
    >>> get_nearest_wmtimes(t0, 3)
     (dt.datetime(2020, 1, 1, 9, 0), dt.datetime(2020, 1, 1, 12, 0))
    """
    # get the closest time available
    tclose = round_time(t0, roundTo=time_delta * 60 * 60)

    # Just calculate both options and take the closest
    t2_1 = tclose + dt.timedelta(hours=time_delta)
    t2_2 = tclose - dt.timedelta(hours=time_delta)
    t2 = [t2_1 if get_dt(t2_1, t0) < get_dt(t2_2, t0) else t2_2][0]

    # If you're within 5 minutes just take the closest time
    if get_dt(tclose, t0) < _THRESHOLD_SECONDS:
        return [tclose]
    else:
        if t2 > tclose:
            return [tclose, t2]
        else:
            return [t2, tclose]


def get_dt(t1: dt.datetime, t2: dt.datetime) -> float:
    """
    Helper function for getting the absolute difference in seconds between
    two python datetimes.

    Args:
        t1: Python datetimes
        t2: Python datetimes

    Returns:
        Absolute difference in seconds between the two inputs

    Examples:
    >>> import datetime as dt
    >>> from RAiDER.utilFcns import get_dt
    >>> get_dt(dt.datetime(2020,1,1,5,0,0), dt.datetime(2020,1,1,0,0,0))
     18000.0
    """
    return np.abs((t1 - t2).total_seconds())


# Tell PyYAML how to serialize pathlib Paths
yaml.add_representer(
    pathlib.PosixPath,
    lambda dumper, data: dumper.represent_scalar(
        'tag:yaml.org,2002:str',
        str(data)
    )
)
yaml.add_representer(
    tuple,
    lambda dumper, data: dumper.represent_sequence(
        'tag:yaml.org,2002:seq',
        data
    )
)

def write_yaml(content: dict[str, Any], dst: Union[str, Path]) -> Path:
    """Write a new yaml file from a dictionary with template.yaml as a base.

    Each key-value pair in 'content' will override the one from template.yaml.
    """
    yaml_path = Path(RAiDER.__file__).parent / 'cli/examples/template/template.yaml'

    with yaml_path.open() as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f'Something is wrong with the yaml file {yaml_path}')

    params = {**params, **content}

    dst = Path(dst)
    with dst.open('w') as fh:
        yaml.dump(params, fh, default_flow_style=False)

    logger.info('Wrote new cfg file: %s', str(dst))
    return dst


def parse_crs(proj: CRSLike) -> CRS:
    """Parses through the projections."""
    if isinstance(proj, CRS):
        return proj
    elif isinstance(proj, str):
        return CRS.from_epsg(proj.lstrip('EPSG:'))
    elif isinstance(proj, int):
        return CRS.from_epsg(proj)
    raise TypeError(f'Data type "{type(proj)}" not supported for CRS')
