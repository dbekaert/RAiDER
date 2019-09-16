import datetime
import numpy as np
import os
import pyproj
import re

from RAiDER.models.weatherModel import WeatherModel


def Model():
    return HRRR()

class HRRR(WeatherModel):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'pl' # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'hrrr'
        self._dataset = 'hrrr'

        # Tuple of min/max years where data is available. 
        self._valid_range = (datetime.date(2018,7,15),)
        self._lag_time = datetime.timedelta(hours=3) # Availability lag time in days

        # model constants: TODO: need to update/double-check these
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233 # [K/Pa]
        self._k3 = 3.75e3 # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 3./111
        self._lon_res = 3./111
        self._x_res = 3.
        self._y_res = 3.

        self._Nproc = 1
        self._Name = 'HRRR'
        self._Npl = 0
        self._files = None
        self._bounds = None

        # Projection
        # See https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb and code lower down
        # '262.5:38.5:38.5:38.5 237.280472:1799:3000.00 21.138123:1059:3000.00'
        # 'lov:latin1:latin2:latd lon1:nx:dx lat1:ny:dy'
        # LCC parameters
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        p1 = pyproj.Proj(proj='lcc', lat_1=lat1,
                             lat_2=lat2, lat_0=lat0,
                             lon_0=lon0, a=6370, b=6370,
                             towgs84=(0,0,0), no_defs=True)
        self._proj = p1

    def fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch weather model data from HRRR
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)
        self._files = download_hrrr_file(time, 'hrrr', out = out, field = 'prs', verbose = True)

    def load_weather(self, filename = None, bounds = None):
        '''
        Load a weather model into a python weatherModel object, from self._files if no
        filename is passed. 
        '''
        if filename is None:
            filename = self._files

        # read data from grib file
        xArr, yArr, lats, lons, temps, qs, geo_hgt, pl = \
                      makeDataCubes(filename, verbose = False)

        Ny, Nx = lats.shape

        lons[lons > 180] -= 360

        # data cube format should be lons, lats, heights
        _xs = np.broadcast_to(xArr[:, np.newaxis, np.newaxis],
                                     geo_hgt.shape)
        _ys = np.broadcast_to(yArr[np.newaxis, :, np.newaxis],
                                     geo_hgt.shape)
        _lons = np.broadcast_to(lons[..., np.newaxis],
                                     geo_hgt.shape)
        _lats = np.broadcast_to(lats[..., np.newaxis],
                                     geo_hgt.shape)

        # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs

        self._p = np.broadcast_to(pl[np.newaxis, np.newaxis, :],
                                  self._zs.shape)
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons

        if self._bounds is not None:
            lat_min, lat_max, lon_min, lon_max = self._bounds
            self._restrict_model(lat_min, lat_max, lon_min, lon_max)


def makeDataCubes(outName, verbose = False):
    '''
    Create a cube of data representing temperature and relative humidity 
    at specified pressure levels    
    '''
    pl = getPresLevels()
    pl = np.array([convertmb2Pa(p) for p in pl['Values']])

    t, z, q, xArr, yArr, lats, lons = pull_hrrr_data(outName, verbose = verbose)

    return xArr, yArr, lats.T, lons.T, np.moveaxis(t, [0,1,2], [2, 1, 0]), np.moveaxis(q, [0,1,2], [2, 1, 0]), np.moveaxis(z, [0,1,2], [2, 1, 0]), pl


def convertmb2Pa(pres):
    return 100*pres


def getPresLevels():
    presList = [int(v) for v in range(50, 1025, 25)]
    presList.append(1013.2)
    outDict = {'Values': presList, 'units': 'mb', 'Name': 'Pressure_levels'}
    return outDict


def pull_hrrr_data(filename, verbose = False):
    '''
    Get the variables from a HRRR grib2 file
    '''
    from cfgrib.xarray_store import open_dataset

    # Pull the native grid
    xArr, yArr = self.getXY_gdal(filename)

    # open the dataset and pull the data
    ds = open_dataset(filename,
         backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}})
    t = ds['t'].values.copy()
    z = ds['gh'].values.copy()
    q = ds['q'].values.copy()
    lats = ds['t'].latitude.values.copy()
    lons = ds['t'].longitude.values.copy()

    del ds

    return t, z, q, xArr, yArr, lats, lons


def download_hrrr_file(DATE, model, out, field = 'prs', verbose = False):
    ''' 
    Download a HRRR model
    ''' 
    import requests

    fxx = '00'
    outfile = '{}_{}_{}_f00.grib2'.format(model, DATE.strftime('%Y%m%d_%H%M%S'), field)

    grib2file = 'https://pando-rgw01.chpc.utah.edu/{}/{}/{}/{}.t{:02d}z.wrf{}f{}.grib2' \
                    .format(model, field,  DATE.strftime('%Y%m%d'), model, DATE.hour, field, fxx)

    if verbose:
       print('Downloading {} to {}'.format(grib2file, out))

    r = requests.get(grib2file)
    with open(out, 'wb') as f:
       f.write(r.content)

    if verbose:
       print('Success!')

    return out 


