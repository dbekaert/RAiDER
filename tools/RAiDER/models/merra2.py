import os
import xarray

import datetime as dt
import numpy as np
import pydap.cas.urs
import pydap.client

from pyproj import CRS

from RAiDER.models.weatherModel import WeatherModel
from RAiDER.logger import logger
from RAiDER.utilFcns import writeWeatherVarsXarray, read_EarthData_loginInfo
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)


# Path to Netrc file, can be controlled by env var
# Useful for containers - similar to CDSAPI_RC
EARTHDATA_RC = os.environ.get("EARTHDATA_RC", None)


def Model():
    return MERRA2()


class MERRA2(WeatherModel):
    # I took this from MERRA-2 model level weblink
    # https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/M2I3NVASM.5.12.4/
    def __init__(self):

        import calendar

        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'ml'  # Default, pressure levels are 'pl'

        self._classname = 'merra2'
        self._dataset = 'merra2'

        # Tuple of min/max years where data is available.
        utcnow = dt.datetime.utcnow()
        enddate = dt.datetime(utcnow.year, utcnow.month, 15) - dt.timedelta(days=60)
        enddate = dt.datetime(enddate.year, enddate.month, calendar.monthrange(enddate.year, enddate.month)[1])
        self._valid_range = (dt.datetime(1980, 1, 1), "Present")
        lag_time = utcnow - enddate
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

    def _fetch(self, out):
        '''
        Fetch weather model data from GMAO: note we only extract the lat/lon bounds for this weather model; fetching data is not needed here as we don't actually download any data using OpenDAP
        '''
        time = self._time 
        
        # check whether the file already exists
        if os.path.exists(out):
            return

        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._ll_bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._ll_bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._ll_bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._ll_bounds[3] - (-180.0)) / self._lon_res)

        lats = np.arange(
            (-90 + lat_min_ind * self._lat_res),
            (-90 + (lat_max_ind + 1) * self._lat_res),
            self._lat_res
        )
        lons = np.arange(
            (-180 + lon_min_ind * self._lon_res),
            (-180 + (lon_max_ind + 1) * self._lon_res),
            self._lon_res
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

        T0 = dt.datetime(time.year, time.month, time.day, 0, 0, 0)
        DT = time - T0
        time_ind = int(DT.total_seconds() / 3600.0 / 3.0)

        # Earthdata credentials
        earthdata_usr, earthdata_pwd = read_EarthData_loginInfo(EARTHDATA_RC)

        # open the dataset and pull the data
        url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/opendap/MERRA2/M2T3NVASM.5.12.4/' + time.strftime('%Y/%m') + '/MERRA2_' + str(url_sub) + '.tavg3_3d_asm_Nv.' + time.strftime('%Y%m%d') + '.nc4'

        session = pydap.cas.urs.setup_session(earthdata_usr, earthdata_pwd, check_url=url)
        stream = pydap.client.open_url(url, session=session)

        q = stream['QV'][0,:,lat_min_ind:lat_max_ind + 1, lon_min_ind:lon_max_ind + 1].data.squeeze()
        p = stream['PL'][0,:,lat_min_ind:lat_max_ind + 1, lon_min_ind:lon_max_ind + 1].data.squeeze()
        t = stream['T'][0,:,lat_min_ind:lat_max_ind + 1, lon_min_ind:lon_max_ind + 1].data.squeeze()
        h = stream['H'][0,:,lat_min_ind:lat_max_ind + 1, lon_min_ind:lon_max_ind + 1].data.squeeze()

        try:
            writeWeatherVarsXarray(lat, lon, h, q, p, t, time, self._proj, outName=out)
        except Exception as e:
            logger.debug(e)
            logger.exception("MERRA-2: Unable to save weathermodel to file")
            raise RuntimeError('MERRA-2 failed with the following error: {}'.format(e))

    def load_weather(self,  f=None, *args, **kwargs):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        f = self.files[0] if f is None else f
        self._load_model_level(f)

    def _load_model_level(self, filename):
        '''
        Get the variables from the GMAO link using OpenDAP
        '''

        # adding the import here should become absolute when transition to netcdf
        ds = xarray.load_dataset(filename)
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
