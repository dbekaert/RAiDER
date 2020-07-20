import datetime
import numpy as np
import os
from pyproj import CRS
import re

from RAiDER.models.weatherModel import WeatherModel


def Model():
    return GMAO()

class GMAO(WeatherModel):
    # I took this from GMAO model level weblink
    # https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'ml' # Default, pressure levels are 'pl'
        
        self._classname = 'gmao'
        self._dataset = 'gmao'

        # Tuple of min/max years where data is available. 
        self._valid_range = (datetime.datetime(2017,12,1),"Present")
        self._lag_time = datetime.timedelta(hours=0.3125) # Availability lag time in days

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233 # [K/Pa]
        self._k3 = 3.75e3 # [K^2/Pa]

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

    
    
    
    def _fetch(self, lats, lons, time, out, Nextra = 2):
        '''
        Fetch weather model data from GMAO: note we only extract the lat/lon bounds for this weather model; fetching data is not needed here as we don't actually download any data using OpenDAP
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)


    
    

    def load_weather(self,f):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        self._load_model_level(f)


    
    def _load_model_level(self,f):
        '''
        Get the variables from the GMAO link using OpenDAP
        '''
        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._bounds[0] - (-90.0)) / self._lat_res)
        lat_max_ind = int((self._bounds[1] - (-90.0)) / self._lat_res)
        lon_min_ind = int((self._bounds[2] - (-180.0)) / self._lon_res)
        lon_max_ind = int((self._bounds[3] - (-180.0)) / self._lon_res)
        
        import datetime as dt
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
        
        q = ds['qv'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        p = ds['pl'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        t = ds['t'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        h = ds['h'].array[time_ind,ml_min:(ml_max+1),lat_min_ind:(lat_max_ind+1),lon_min_ind:(lon_max_ind+1)][0]
        
        # calculate the lat, lon and mean h for each layer in the regular grid
        hs = np.zeros(72)
        for ii in range(72):
            hs[ii] = np.mean(h[ii,:,:])
        
        lats = np.arange((-90 + lat_min_ind * self._lat_res), (-90 + (lat_max_ind+1) * self._lat_res), self._lat_res)
        lons = np.arange((-180 + lon_min_ind * self._lon_res), (-180 + (lon_max_ind+1) * self._lon_res), self._lon_res)
        
        # restructure the 3-D lat/lon/h in regular grid
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],t.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],t.shape)
        _hs = np.broadcast_to(hs[:, np.newaxis, np.newaxis],t.shape)
        
        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = np.transpose(p)
        q = np.transpose(q)
        t = np.transpose(t)
        h = np.transpose(h)
        _lats = np.transpose(_lats)
        _lons = np.transpose(_lons)
        _hs = np.transpose(_hs)
        
        
        # check this
        # data cube format should be lats,lons,heights
        p = p.swapaxes(0,1)
        q = q.swapaxes(0,1)
        t = t.swapaxes(0,1)
        h = h.swapaxes(0,1)
        _lats = _lats.swapaxes(0,1)
        _lons = _lons.swapaxes(0,1)
        _hs = _hs.swapaxes(0,1)
        
        
        # For some reason z is opposite the others
        p = np.flip(p, axis = 2)
        q = np.flip(q, axis = 2)
        t = np.flip(t, axis = 2)
        h = np.flip(h, axis = 2)
        _lats = np.flip(_lats, axis = 2)
        _lons = np.flip(_lons, axis = 2)
        _hs = np.flip(_hs, axis = 2)
        

        # interpolate (p,q,t) along the vertical axis from irregular grid to regular grid
        from RAiDER.interpolator import interp_along_axis
        p_intrpl = interp_along_axis(h, _hs, p, axis = 2)
        q_intrpl = interp_along_axis(h, _hs, q, axis = 2)
        t_intrpl = interp_along_axis(h, _hs, t, axis = 2)
        


        
        # assign the regular-grid (lat/lon/h) variables
        
        self._p = p_intrpl
        self._q = q_intrpl
        self._t = t_intrpl
        self._lats = _lats
        self._lons = _lons
        self._xs = _lons
        self._ys = _lats
        self._zs = _hs
        
 
