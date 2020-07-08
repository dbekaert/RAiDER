import datetime
import numpy as np
import os
from pyproj import CRS
import re

from RAiDER.models.weatherModel import WeatherModel


def Model():
    return GMAO()

class GMAO(WeatherModel):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'ml' # Default, pressure levels are 'pl'
        
        self._classname = 'gmao'
        self._dataset = 'gmao'

        # Tuple of min/max years where data is available. 
        self._valid_range = (datetime.datetime(2017,12,01),"Present")
        self._lag_time = datetime.timedelta(hours=0.3125) # Availability lag time in days

        # model constants: TODO: need to update/double-check these
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233 # [K/Pa]
        self._k3 = 3.75e3 # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 0.25
        self._lon_res = 0.3125
        self._x_res = 0.3125
        self._y_res = 0.25

        self._Name = 'GMAO'
        self._files = None
        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch weather model data from GMAO
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)
#        self._files = self._download_hrrr_file(time, 'hrrr', out = out,
#                                          field = 'prs', verbose = True)

    
    

    def load_weather(self):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        self._load_model_level()


    
    def _load_model_level(self):
        '''
        Get the variables from a HRRR grib2 file
        '''



        lat_min_ind = int((self._bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._bounds[3] - (-180.0)) / self._lon_res)
        
        from datetime import datetime as dt
        T0 = dt.datetime(2017,12,1,0,0,0)
        DT = self._time - T0
        time_ind = int(DT.total_seconds() / 3600.0 / 3.0)
        
        ml_min = 0
        ml_max = 71

        # open the dataset and pull the data
        import pydap.client
        import pydap.cas.urs
        url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv'
        session = pydap.cas.urs.setup_session('username', 'password', check_url=url)
        ds = pydap.client.open_url(url, session=session)
        t = ds['t'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        q = ds['qv'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        p = ds['pl'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        z = ds['h'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        
        lats = np.arange((-90 + lat_min_ind * self._lat_res), (-90 + (lat_max_ind+1) * self._lat_res), self._lat_res)
        lons = np.arange((-90 + lon_min_ind * self._lon_res), (-90 + (lon_max_ind+1) * self._lon_res), self._lon_res)
        
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],t.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],t.shape)
        
        self._p = p
        self._q = q
        self._t = t
        self._lats = _lats
        self._lons = _lons
        self._xs = self._lons.copy()
        self._ys = self._lats.copy()
        self._zs = z
        
        
        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p)
        self._t = np.transpose(self._t)
        self._q = np.transpose(self._q)
        self._lats = np.transpose(self._lats)
        self._lons = np.transpose(self._lons)
        self._ys = np.transpose(self._ys)
        self._xs = np.transpose(self._xs)
        self._zs = np.transpose(self._zs)
        
        # check this
        # data cube format should be lats,lons,heights
        self._lats = self._lats.swapaxes(0,1)
        self._lons = self._lons.swapaxes(0,1)
        self._xs = self._xs.swapaxes(0,1)
        self._ys = self._ys.swapaxes(0,1)
        self._zs = self._zs.swapaxes(0,1)
        self._p = self._p.swapaxes(0,1)
        self._q = self._q.swapaxes(0,1)
        self._t = self._t.swapaxes(0,1)
        
        # For some reason z is opposite the others
        self._p = np.flip(self._p, axis = 2)
        self._t = np.flip(self._t, axis = 2)
        self._q = np.flip(self._q, axis = 2)
        self._xs = np.flip(self._xs, axis = 2)
        self._ys = np.flip(self._ys, axis = 2)
        self._zs = np.flip(self._zs, axis = 2)
        self._lats = np.flip(self._lats, axis = 2)
        self._lons = np.flip(self._lons, axis = 2)
        



