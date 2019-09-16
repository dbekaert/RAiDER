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
        self._files = self._download_hrrr_file(time, 'hrrr', out = out, 
                                          field = 'prs', verbose = True)

    def _getPresLevels(self, low = 50, high = 1013.2, inc = 25):
        presList = [float(v) for v in range(int(low//1), int(high//1), int(inc//1))]
        presList.append(high)
        outDict = {'Values': presList, 'units': 'mb', 'Name': 'Pressure_levels'}
        return outDict

    def _download_hrrr_file(self, DATE, model, out, field = 'prs', verbose = False):
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
