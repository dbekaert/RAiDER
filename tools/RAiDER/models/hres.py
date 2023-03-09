import datetime

import numpy as np

from pyproj import CRS

from RAiDER.models.ecmwf import ECMWF
from RAiDER.models.weatherModel import WeatherModel, TIME_RES
from RAiDER.models.model_levels import (
    LEVELS_91_HEIGHTS,
    LEVELS_25_HEIGHTS,
    A_91_HRES,
    B_91_HRES,
)


class HRES(ECMWF):
    '''
    Implement ECMWF models
    '''

    def __init__(self, level_type='ml'):
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]


        # 9 km horizontal grid spacing. This is only used for extending the download-buffer, i.e. not in subsequent processing.
        self._lon_res = 9. / 111  # 0.08108115
        self._lat_res = 9. / 111  # 0.08108115
        self._x_res = 9. / 111  # 0.08108115
        self._y_res = 9. / 111  # 0.08108115

        self._humidityType = 'q'
        # Default, pressure levels are 'pl'
        self._expver = '1'
        self._classname = 'od'
        self._dataset = 'hres'
        self._Name = 'HRES'
        self._proj = CRS.from_epsg(4326)

        self._time_res = TIME_RES[self._dataset.upper()]
        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(1983, 4, 20), "Present")
        # Availability lag time in days
        self._lag_time = datetime.timedelta(hours=6)

        self.setLevelType('ml')

    def update_a_b(self):
        # Before 2013-06-26, there were only 91 model levels. The mapping coefficients below are extracted
        # based on https://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
        self._levels = 91
        self._zlevels = np.flipud(LEVELS_91_HEIGHTS)
        self._a = A_91_HRES
        self._b = B_91_HRES

    def load_weather(self, filename=None):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        if filename is None:
            filename = self.files[0]

        if self._model_level_type == 'ml':
            if (self._time < datetime.datetime(2013, 6, 26, 0, 0, 0)):
                self.update_a_b()
            self._load_model_level(filename)
        elif self._model_level_type == 'pl':
            self._load_pressure_levels(filename)

    def _fetch(self,out):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._ll_bounds
        time = self._time

        if (time < datetime.datetime(2013, 6, 26, 0, 0, 0)):
            self.update_a_b()

        # execute the search at ECMWF
        self._download_ecmwf(lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time, out)
