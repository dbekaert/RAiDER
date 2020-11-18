import datetime as dt
import numpy as np
import pandas as pd
import shutil
import h5py
import pydap.cas.urs
import pydap.client
import requests
from pyproj import CRS

from RAiDER.models.weatherModel import WeatherModel
from RAiDER.logger import *
from RAiDER.utilFcns import writeWeatherVars2HDF5, roundTime


class GMAO(WeatherModel):
    # I took this from GMAO model level weblink
    # https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._model_file_type = 'h5'

        self._humidityType = 'q'
        self._model_level_type = 'ml'  # Default, pressure levels are 'pl'

        self._classname = 'gmao'
        self._dataset = 'gmao'

        # Tuple of min/max years where data is available.
        self._valid_range = (dt.datetime(2014, 2, 20), "Present")
        self._lag_time = dt.timedelta(hours=24.0)  # Availability lag time in hours

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # horizontal grid spacing
        self._lat_res = 0.25
        self._lon_res = 0.3125
        self._x_res = 0.3125
        self._y_res = 0.25

        self._Name = 'GMAO'
        self._files = None
        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch weather model data from GMAO
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._bounds[3] - (-180.0)) / self._lon_res)

        T0   = dt.datetime(2017, 12, 1, 0, 0, 0)
        ## round time to nearest third hour
        time1 = time
        time  = roundTime(time, 3*60*60)
        if not time1 == time:
            logger.warning('Rounded given hour from  %d to %d', time1.hour, time.hour)
            
        DT   = time - T0
        time_ind = int(DT.total_seconds() / 3600.0 / 3.0)

        ml_min = 0
        ml_max = 71
        if time >= T0:
            # open the dataset and pull the data
            url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv'
            session = pydap.cas.urs.setup_session('username', 'password', check_url=url)
            ds = pydap.client.open_url(url, session=session)
            q  = ds['qv'].array[
                time_ind,
                ml_min:(ml_max + 1),
                lat_min_ind:(lat_max_ind + 1),
                lon_min_ind:(lon_max_ind + 1)
            ][0]

            p = ds['pl'].array[
                time_ind,
                ml_min:(ml_max + 1),
                lat_min_ind:(lat_max_ind + 1),
                lon_min_ind:(lon_max_ind + 1)
            ][0]
            t = ds['t'].array[
                time_ind,
                ml_min:(ml_max + 1),
                lat_min_ind:(lat_max_ind + 1),
                lon_min_ind:(lon_max_ind + 1)
            ][0]
            h = ds['h'].array[
                time_ind,
                ml_min:(ml_max + 1),
                lat_min_ind:(lat_max_ind + 1),
                lon_min_ind:(lon_max_ind + 1)
            ][0]

        else:
            root = 'https://portal.nccs.nasa.gov/datashare/gmao/geos-fp/das/Y{}/M{:02d}/D{:02d}'
            base = f'GEOS.fp.asm.inst3_3d_asm_Nv.{time.strftime("%Y%m%d")}_{time.hour:02}00.V01.nc4'
            url  = f'{root.format(time.year, time.month, time.day)}/{base}'
            f    = '{}_raw{}'.format(*os.path.splitext(out))
            if not os.path.exists(f):
                logger.info('Fetching URL: %s', url)
                session = requests_retry_session()
                resp    = session.get(url, stream=True)
                assert resp.ok, f'Could not access url for time: {time}'
                with open(f, 'wb') as fh:
                    shutil.copyfileobj(resp.raw, fh)
            else:
                logger.warning('Weather model already exists, skipping download')

            with h5py.File(f, 'r') as ds:
                q = ds['QV'][0, :, lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)]
                p = ds['PL'][0, :, lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)]
                t = ds['T'][0, :, lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)]
                h = ds['H'][0, :, lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)]

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

        try:
            # Note that lat/lon gets written twice for GMAO because they are the same as y/x
            writeWeatherVars2HDF5(lats, lons, lons, lats, h, q, p, t, self._proj, )
        except Exception:
            logger.exception("Unable to save weathermodel to file")


    def load_weather(self, f):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        self._load_model_level(f)


    def _load_model_level(self, filename):
        '''
        Get the variables from the GMAO link using OpenDAP
        '''

        # adding the import here should become absolute when transition to netcdf
        import h5py
        with h5py.File(filename, 'r') as f:
            lons = f['lons'][:].copy()
            lats = f['lats'][:].copy()
            h = f['z'][:].copy()
            p = f['p'][:].copy()
            q = f['q'][:].copy()
            t = f['t'][:].copy()


        # restructure the 3-D lat/lon/h in regular grid
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :], t.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis], t.shape)

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = np.transpose(p)
        q = np.transpose(q)
        t = np.transpose(t)
        h = np.transpose(h)
        _lats = np.transpose(_lats)
        _lons = np.transpose(_lons)

        # check this
        # data cube format should be lats,lons,heights
        p = p.swapaxes(0, 1)
        q = q.swapaxes(0, 1)
        t = t.swapaxes(0, 1)
        h = h.swapaxes(0, 1)
        _lats = _lats.swapaxes(0, 1)
        _lons = _lons.swapaxes(0, 1)

        # For some reason z is opposite the others
        p = np.flip(p, axis=2)
        q = np.flip(q, axis=2)
        t = np.flip(t, axis=2)
        h = np.flip(h, axis=2)
        _lats = np.flip(_lats, axis=2)
        _lons = np.flip(_lons, axis=2)

        # assign the regular-grid (lat/lon/h) variables

        self._p = p
        self._q = q
        self._t = t
        self._lats = _lats
        self._lons = _lons
        self._xs = _lons
        self._ys = _lats
        self._zs = h

def requests_retry_session(retries=10, session=None):
    """ https://www.peterbe.com/plog/best-practice-with-retries-with-requests """
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    # add a retry strategy; https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/
    session = session or requests.Session()
    retry   = Retry(total=retries, read=retries, connect=retries,
                    backoff_factor=0.3, status_forcelist=list(range(429, 505)))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session
