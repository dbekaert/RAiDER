from datetime import datetime, timedelta
import os
import re

import numpy as np
from pyproj import CRS

from RAiDER.interpolator import interp_along_axis
from RAiDER.models.weatherModel import WeatherModel


class CustomWeatherModel(WeatherModel):

    # General model initiation
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        # Name of the model
        self._Name = 'CustomWeatherModel'

        # Humidity can be specified as either specific ('q') or relative humidity ('rh')
        self._humidityType = 'q'  # Can be 'q' for specific humidity or 'rh' for relative humidity

        # Your weather model might be specified at fixed pressure levels (pl) or on generic 
        # 'model levels' (ml).
        self._model_level_type = 'ml'  # Default, for fixed pressure levels use 'pl'.

        # Most weather models have a range of dates when it is available
        # self._valid_range = (start_time, end_time)
        self._valid_range = (datetime(2017, 12, 1), "Present")

        # Most weather models have a lag time until they are available, so add yours
        self._lag_time = timedelta(hours=7.5)

        # Specify horizontal grid spacing in both the native grid spacing and lat/lon
        '''
        Example:
            self._lat_res = 3. / 111
            self._lon_res = 3. / 111
            self._x_res = 3.
            self._y_res = 3.
        '''

        '''
        # RAiDER needs the native projection of the weather model specified in a pyproj (PROJ.4)-compatible form
        Example: a Lambert conformal conic projection used by HRRR
            self._proj = CRS('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} +x_0={x0} +y_0={y0} +a={a} +b={a} +units=m +no_defs'.format(lat1=lat1, lat2=lat2, lat0=lat0, lon0=lon0, x0=x0, y0=y0, a=earth_radius))

        Example: WGS84 
            self._proj = CRS.from_epsg(4326)
        '''

    def _fetch(self, time, *args, **kwargs):
        #TODO: Decide what fetch and load should each do.
        '''
        The "_fetch" method is used to download data from a server or read data from a file
        Specify variables that are needed to download/read the data, e.g. a bounding box. 
        
        ######################################################################
        Example calling helper functions: 
            # You can specify helper functions that do specific tasks as class methods. 
            # Include these after the other class definitions
            bounds = self._generic_helper_function(*args, **kwargs)
            self._files = self._download_file(*args, **kwargs)

        ######################################################################
        # Example reading data using PyDAP. 
        # See the gmao.py module for more details
            import pydap.cas.urs
            import pydap.client

            # open the dataset and pull the data
            url = 'https://opendap.nccs.nasa.gov/dods/GEOS-5/fp/0.25_deg/assim/inst3_3d_asm_Nv'

            session = pydap.cas.urs.setup_session(
                'username', 'password', check_url=url)
            ds = pydap.client.open_url(url, session=session)
            self._q = ds['qv'].array[
                time_ind, 
                ml_min:(ml_max + 1), 
                lat_min_ind:(lat_max_ind + 1), 
                lon_min_ind:(lon_max_ind + 1)
            ][0]

        ######################################################################
        # Example using GRIB files and Xarray. See the hrrr.py module for more details
            from cfgrib.xarray_store import open_dataset
            filename = 'my_netcdf_file_containing_weather_model_variables'
            ds = open_dataset(
                filename,
                backend_kwargs={'filter_by_keys': {'typeOfLevel': 'isobaricInhPa'}}
            )
            self._q = ds['q'].values.copy()
            self._lats = ds['q'].latitude.values.copy()
            self._lons = ds['q'].longitude.values.copy()
        ######################################################################
        '''


    def load_weather(self, f):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, the variables _xs, _ys, _zs, _p, _q/_rh,
        _t) should be fully populated.

        This is a generic method that will be run every time you run a new model in 
        RAiDER. You can load and process a downloaded file or process parameters created 
        using the "fetch" method. You 

        You can define different methods for model levels and pressure levels, if your 
        weather model handles both.
        '''
        if self._model_level_type=='ml':
            self._load_model_level(f)
        elif self._model_level_type=='pl':
            self._load_pressure_level(f)
        else:
            raise RuntimeError('Model level Type {} not supported'.format(self._model_level_type))

    # Example helper function to load model levels
    def _load_model_level(self, f):
        '''
        Read data from a server or a local file into a set of variables.
        Required variables:
            t       - temperature specified at nodes of a cube
            q/rh    - specific/relative humidity specified at nodes of a cube
            p       - pressure specified at nodes of a cube
            x/y/z   - Either x/y/z proper or could also be lons/lats/heights in 
                      the native weather model grid

        Note that some variables might go by different names in different models. 
        E.g. a height variable might be called "z", "h", etc. RAiDER expects the 
        variables self._xs, self._ys, self._zs to contain your x/y/z variables, 
        and the self._lats/self._lons to contain your lat/lon variables. 
        '''

        # RAiDER needs the following variables as 3D numpy arrays
        self._q  = np.empty((1,1,1)) 
        self._t  = np.empty((1,1,1)) 
        self._rh = np.empty((1,1,1)) 
        self._p  = np.empty((1,1,1)) 

        # X/Y/Z variables must be specified in the native model grid coordinate system
        # These should be loaded as 3D arrays, RAiDER will reduce them to 1-D coordinates
        self._xs = np.empty((1,1,1))  
        self._ys = np.empty((1,1,1)) 
        self._zs = np.empty((1,1,1)) 

        # lats and lons are WGS84 (EPSG 4326) grid node locations.
        # These will get broadcast to 3D arrays if the model supplies 2D arrays
        self._lats = np.empty((1,1))
        self._lons = np.empty((1,1))

        # RAiDER wants lats and lons for each grid node, even if they are the same at 
        # different heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :], t.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis], t.shape)
        _hs = np.broadcast_to(hs[:, np.newaxis, np.newaxis], t.shape)

        # Every variable must be in (lons, lats, heights) shape. If they come differently, 
        # e.g., (lats, lons, heights), then re-structure everything to match. 
        self._p = np.transpose(p)
        self._q = np.transpose(q)
        self._t = np.transpose(t)
        self._h = np.transpose(h)
        self._lats = np.transpose(_lats)
        self._lons = np.transpose(_lons)
        self._z = np.transpose(_hs)

        # z-values should increase with index (i.e. get higher as the index 
        # in the 3rd dimension gets larger). If your model does not, flip it.
        self._p = np.flip(p, axis=2)
        self._q = np.flip(q, axis=2)
        self._t = np.flip(t, axis=2)
        self._h = np.flip(h, axis=2)
        self._lats = np.flip(_lats, axis=2)
        self._lons = np.flip(_lons, axis=2)
        self._hs = np.flip(_hs, axis=2)

        # If the weather model is natively on a grid other than WGS84 (EPSG 4326),
        # you will need to specify the native x and y coordinates
        self._xs = _lons
        self._ys = _lats

