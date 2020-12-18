import h5py

import datetime as dt
import numpy as np
import pydap.cas.urs
import pydap.client

from pyproj import CRS

from RAiDER.models.weatherModel import WeatherModel
from RAiDER.logger import *
from RAiDER.utilFcns import writeWeatherVars2NETCDF4

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
        self._files = None
        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch weather model data from GMAO: note we only extract the lat/lon bounds for this weather model; fetching data is not needed here as we don't actually download any data using OpenDAP
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        # check whether the file already exists
        if os.path.exists(out):
           return

        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._bounds[3] - (-180.0)) / self._lon_res)

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

        ml_min = 0
        ml_max = 71

        # open the dataset and pull the data
        url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov:443/opendap/MERRA2/M2I3NVASM.5.12.4/' + time.strftime('%Y/%m') + '/MERRA2_' + str(url_sub) + '.inst3_3d_asm_Nv.' + time.strftime('%Y%m%d') + '.nc4'
        session = pydap.cas.urs.setup_session('username', 'password', check_url=url)
        ds = pydap.client.open_url(url, session=session)

        q = ds['QV'][time_ind, ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)][0]
        p = ds['PL'][time_ind, ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)][0]
        t = ds['T'][time_ind, ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)][0]
        h = ds['H'][time_ind, ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)][0]

        try:
            writeWeatherVars2NETCDF4(self, lats, lons, h, q, p, t, outName=out)
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
        from netCDF4 import Dataset
        with Dataset(filename, mode='r') as f:
            lons = np.array(f.variables['x'][:])
            lats = np.array(f.variables['y'][:])
            h = np.array(f.variables['H'][:])
            q = np.array(f.variables['QV'][:])
            p = np.array(f.variables['PL'][:])
            t = np.array(f.variables['T'][:])

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
