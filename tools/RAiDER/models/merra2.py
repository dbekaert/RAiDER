import datetime as dt
import os
from pathlib import Path

import numpy as np
import xarray as xr
from pyproj import CRS

from RAiDER.logger import logger
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.utilFcns import write_weather_vars_to_ds


# Path to Netrc file, can be controlled by env var
# Useful for containers - similar to CDSAPI_RC
EARTHDATA_RC = os.environ.get('EARTHDATA_RC', None)


def Model():
    return MERRA2()


class MERRA2(WeatherModel):
    def __init__(self) -> None:
        import calendar

        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'ml'  # Default, pressure levels are 'pl'

        self._classname = 'merra2'
        self._dataset = 'merra2'

        # Tuple of min/max years where data is available.
        utcnow = dt.datetime.now(dt.timezone.utc)
        enddate = dt.datetime(utcnow.year, utcnow.month, 15) - dt.timedelta(days=60)
        enddate = dt.datetime(enddate.year, enddate.month, calendar.monthrange(enddate.year, enddate.month)[1])
        self._valid_range = (
            dt.datetime(1980, 1, 1).replace(tzinfo=dt.timezone(offset=dt.timedelta())),
            dt.datetime.now(dt.timezone.utc),
        )
        lag_time = utcnow - enddate.replace(tzinfo=dt.timezone(offset=dt.timedelta()))
        self._lag_time = dt.timedelta(days=lag_time.days)  # Availability lag time in days
        self._time_res = 1

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # horizontal grid spacing
        self._lat_res = 0.5
        self._lon_res = 0.625
        self._x_res = 0.625
        self._y_res = 0.5

        self._Name = 'MERRA2'
        self.files = None
        self._bounds = None
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, out: Path) -> None:
        """Fetch weather model data from GMAO.
        
        Note: we only extract the lat/lon bounds for this weather model; fetching data is not needed here as we don't
        actually download any data using OPeNDAP.
        """
        time = self._time

        # check whether the file already exists
        if out.exists():
            return

        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._ll_bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._ll_bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._ll_bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._ll_bounds[3] - (-180.0)) / self._lon_res)

        lats = np.arange((-90 + lat_min_ind * self._lat_res), (-90 + (lat_max_ind + 1) * self._lat_res), self._lat_res)
        lons = np.arange(
            (-180 + lon_min_ind * self._lon_res), (-180 + (lon_max_ind + 1) * self._lon_res), self._lon_res
        )

        lon, lat = np.meshgrid(lons, lats)

        if time.year < 1992:
            url_sub = 100
        elif time.year < 2001:
            url_sub = 200
        elif time.year < 2011:
            url_sub = 300
        else:
            url_sub = 400

        # open the dataset and pull the data
        url = (
            f'dap4://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T3NVASM.5.12.4/'
            f'{time.strftime('%Y/%m')}/'
            f'MERRA2_{url_sub}.tavg3_3d_asm_Nv.{time.strftime('%Y%m%d')}.nc4'
        )

        # pydap engine required for password-protected datasets
        ds = xr.open_dataset(url, decode_times=False, engine='pydap')

        q = ds['QV'][
            0,
            :,
            lat_min_ind : lat_max_ind + 1,
            lon_min_ind : lon_max_ind + 1,
        ].data.squeeze()
        p = ds['PL'][
            0,
            :,
            lat_min_ind : lat_max_ind + 1,
            lon_min_ind : lon_max_ind + 1,
        ].data.squeeze()
        t = ds['T'][
            0,
            :,
            lat_min_ind : lat_max_ind + 1,
            lon_min_ind : lon_max_ind + 1,
        ].data.squeeze()
        h = ds['H'][
            0,
            :,
            lat_min_ind : lat_max_ind + 1,
            lon_min_ind : lon_max_ind + 1,
        ].data.squeeze()

        try:
            write_weather_vars_to_ds(lat, lon, h, q, p, t, time, self._proj, out_path=out)
        except Exception as e:
            logger.debug(e)
            logger.exception('MERRA-2: Unable to save weather model query to file')
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

    def _load_model_level(self, filename) -> None:
        """Get the variables from the GMAO link using OPeNDAP."""
        # adding the import here should become absolute when transition to netcdf
        ds = xr.load_dataset(filename)
        lons = ds['longitude'].values
        lats = ds['latitude'].values
        h = ds['h'].values
        q = ds['q'].values
        p = ds['p'].values
        t = ds['t'].values

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
