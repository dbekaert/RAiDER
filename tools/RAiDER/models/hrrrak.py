import datetime
import os
import rioxarray
import xarray

import numpy as np

from herbie import Herbie
from pathlib import Path
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

from RAiDER.utilFcns import round_date, transform_coords, rio_profile, rio_stats
from RAiDER.models.weatherModel import (
    WeatherModel, TIME_RES
)
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)

from RAiDER.models.hrrr import *


class HRRRAK(WeatherModel):
    def __init__(self):
        # The HRRR-AK model has a few different parameters than HRRR-CONUS.
        # These will get used if a user requests a bounding box in Alaska
        super().__init__()

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 3. / 111
        self._lon_res = 3. / 111
        self._x_res = 3.
        self._y_res = 3.

        self._Nproc = 1
        self._Npl = 0
        self.files = None
        self._bounds = None
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)

        self._classname = 'hrrrak'
        self._dataset = 'hrrrak'
        self._Name = "HRRR-AK"
        self._time_res = TIME_RES['HRRR-AK']
        self._valid_range = (datetime.datetime(2018, 7, 13), "Present")
        self._lag_time = datetime.timedelta(hours=3)
        self._valid_bounds =  Polygon(((195, 40), (157, 55), (175, 70), (260, 77), (232, 52)))

        # The projection information gets read directly from the  weather model file but we
        # keep this here for object instantiation.
        self._proj = CRS.from_string(
            '+proj=stere +ellps=sphere +a=6371229.0 +b=6371229.0 +lat_0=90 +lon_0=225.0 ' +
            '+x_0=0.0 +y_0=0.0 +lat_ts=60.0 +no_defs +type=crs'
        )


    def _fetch(self, out):
        bounds = self._ll_bounds.copy()
        bounds[2:] = np.mod(bounds[2:], 360)

        corrected_DT = round_date(self._time, datetime.timedelta(hours=self._time_res))
        if not corrected_DT == self._time:
            print('Rounded given datetime from {} to {}'.format(self._time, corrected_DT))

        self.checkTime(corrected_DT)
        download_hrrr_file(bounds, corrected_DT, out, model='hrrrak')


    def load_weather(self, *args, filename=None, **kwargs):
        if filename is None:
            filename = self.files[0] if isinstance(self.files, list) else self.files
        _xs, _ys, _lons, _lats, qs, temps, pl, geo_hgt, proj = load_weather_hrrr(filename)
            # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs
        self._p = np.broadcast_to(pl[np.newaxis, np.newaxis, :], geo_hgt.shape)
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons
        self._proj = proj