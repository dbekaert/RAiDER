import datetime as dt
import shutil
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from RAiDER.logger import logger
from RAiDER.models.model_levels import LEVELS_137_HEIGHTS
from RAiDER.models.weatherModel import TIME_RES, WeatherModel
from RAiDER.utilFcns import requests_retry_session, round_date, write_weather_vars_to_ds


class GMAO(WeatherModel):
    # I took this from GMAO model level weblink
    # https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv
    def __init__(self) -> None:
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'ml'  # Default, pressure levels are 'pl'

        self._classname = 'gmao'
        self._dataset = 'gmao'

        # Tuple of min/max years where data is available.
        self._valid_range = (
            dt.datetime(2014, 2, 20).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
            dt.datetime.now(dt.timezone.utc),
        )
        self._lag_time = dt.timedelta(hours=24.0)  # Availability lag time in hours

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._time_res = TIME_RES[self._dataset.upper()]

        # horizontal grid spacing
        self._lat_res = 0.25
        self._lon_res = 0.3125
        self._x_res = 0.3125
        self._y_res = 0.25

        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)

        self._Name = 'GMAO'
        self.files = None
        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, out: Path) -> None:
        """Fetch weather model data from GMAO."""
        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._ll_bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._ll_bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._ll_bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._ll_bounds[3] - (-180.0)) / self._lon_res)

        T0 = dt.datetime(2017, 12, 1, 0, 0, 0).replace(tzinfo=dt.timezone(offset=dt.timedelta()))
        # round time to nearest third hour
        corrected_DT = round_date(self._time, dt.timedelta(hours=self._time_res))
        if not corrected_DT == self._time:
            logger.warning('Rounded given datetime from  %s to %s', self._time, corrected_DT)

        DT = corrected_DT - T0
        time_ind = int(DT.total_seconds() / 3600.0 / self._time_res)

        ml_min = 0
        ml_max = 71
        if corrected_DT >= T0:
            # open the dataset and pull the data
            url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv'
            with xr.open_dataset(url, decode_times=False) as ds:
                q = ds['qv'][
                    time_ind,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]
                p = ds['pl'][
                    time_ind,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]
                t = ds['t'][
                    time_ind,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]
                h = ds['h'][
                    time_ind,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]

        else:
            root = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{}/M{:02d}/D{:02d}'
            filename = f'GEOS.fp.asm.inst3_3d_asm_Nv.{corrected_DT.strftime("%Y%m%d")}_{corrected_DT.hour:02}00.V01.nc4'
            url = f'{root.format(corrected_DT.year, corrected_DT.month, corrected_DT.day)}/{filename}'
            url += '#mode=bytes'  # https://github.com/pydata/xarray/issues/3653#issuecomment-832712426
            with xr.open_dataset(url) as ds:
                q = ds['QV'][
                    0,  # time (always just 1)
                    ml_min : (ml_max + 1),  # lev
                    lat_min_ind : (lat_max_ind + 1),  # lat
                    lon_min_ind : (lon_max_ind + 1),  # lon
                ]
                p = ds['PL'][
                    0,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]
                t = ds['T'][
                    0,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]
                h = ds['H'][
                    0,
                    ml_min : (ml_max + 1),
                    lat_min_ind : (lat_max_ind + 1),
                    lon_min_ind : (lon_max_ind + 1),
                ]

        lats = np.arange(
            -90 + lat_min_ind * self._lat_res,
            -90 + (lat_max_ind + 1) * self._lat_res,
            self._lat_res,
        )
        lons = np.arange(
            -180 + lon_min_ind * self._lon_res,
            -180 + (lon_max_ind + 1) * self._lon_res,
            self._lon_res,
        )
        lon, lat = np.meshgrid(lons, lats)

        try:
            # Note that lat/lon gets written twice for GMAO because they are the same as y/x
            write_weather_vars_to_ds(lat, lon, h, q, p, t, self._time, self._proj, out)
        except:
            logger.exception('Unable to save weathermodel to file:')
            raise

    def load_weather(self, f=None, *args, **kwargs) -> None:
        """
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        """
        f = self.files[0] if f is None else f
        self._load_model_level(f)

    def _load_model_level(self, filename: Path) -> None:
        """Get the variables from the GMAO link using OPeNDAP."""
        with xr.open_dataset(filename) as ds:
            lons = ds['x']
            lats = ds['y']
            h = ds['h'].data
            q = ds['q'].data
            p = ds['p'].data
            t = ds['t'].data

        # restructure the 1-D lat/lon in regular 2D grid
        lons, lats = np.meshgrid(lons, lats)

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = np.transpose(p)
        q = np.transpose(q)
        t = np.transpose(t)
        h = np.transpose(h)

        # check this
        # data cube format should be lats,lons,heights
        p = p.swapaxes(0, 1)
        q = q.swapaxes(0, 1)
        t = t.swapaxes(0, 1)
        h = h.swapaxes(0, 1)

        # For some reason z is opposite the others
        p = np.flip(p, axis=2)
        q = np.flip(q, axis=2)
        t = np.flip(t, axis=2)
        h = np.flip(h, axis=2)

        # assign the regular-grid (lat/lon/h) variables
        self._p = p
        self._q = q
        self._t = t
        self._lats = lats
        self._lons = lons
        self._xs = lons
        self._ys = lats
        self._zs = h
