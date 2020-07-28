from RAiDER.models.weatherModel import WeatherModel

class customModelReader(WeatherModel):
    def __init__(self):
        WeatherModel.__init__(self)
        self._humidityType = 'q'  # can be "q" (specific humidity) or "rh" (relative humidity)
        self._model_level_type = 'pl'  # Default, pressure levels are "pl", and model levels are "ml"
        self._classname = 'abcd'  # name of the custom weather model
        self._dataset = 'abcd'  # same name as above

        # Tuple of min/max years where data is available.
        #  valid range of the dataset. Users need to specify the start date and end date (can be "present")
        self._valid_range = (datetime.datetime(2016, 7, 15), "Present")
        #  Availability lag time. Can be specified in hours "hours=3" or in days "days=3"
        self._lag_time = datetime.timedelta(hours=3)

        # model constants (these three constants are borrowed from ECMWF model and currently
        # set to be default for all other models, which may need to be double checked.)
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # horizontal grid spacing
        self._lat_res = 3. / 111  # grid spacing in latitude
        self._lon_res = 3. / 111  # grid spacing in longitude
        self._x_res = 3.  # x-direction grid spacing in the native weather model projection
        #  (if the projection is in lat/lon, it is the same as "self._lon_res")
        self._y_res = 3.  # y-direction grid spacing in the weather model native projection
        #  (if the projection is in lat/lon, it is the same as "self._lat_res")

        self._Name = 'ABCD'  # name of the custom weather model (better to be capitalized)

        # Projections in RAiDER are defined using pyproj (python wrapper around Proj)
        # If the projection is defined with EPSG code, one can use "self._proj = CRS.from_epsg(4326)"
        # to replace the following lines to get "self._proj".
        # Below we show the example of HRRR model with the parameters of its Lambert Conformal Conic projection
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        x0 = 0
        y0 = 0
        earth_radius = 6371229
        p1 = CRS('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} +x_0={x0} +y_0={y0} +a={a} +b={a} +units=m +no_defs'.format(lat1=lat1, lat2=lat2, lat0=lat0, lon0=lon0, x0=x0, y0=y0, a=earth_radius))
        self._proj = p1

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch weather model data from the custom weather model "ABCD"
        Inputs (no need to change in the custom weather model reader): 
        lats - latitude 
        lons - longitude 
        time - datatime object (year,month,day,hour,minute,second)
        out - name of downloaded dataset file from the custom weather model server
        Nextra - buffer of latitude/longitude for determining the bounding box 
        '''

        # bounding box plus a buffer using the helper function from the WeatherModel base class
        # This part can be kept without modification.
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        # Auxilliary function:
        # download dataset of the custom weather model "ABCD" from a server and then save it to a file named out.
        # This function needs to be writen by the users. For download from the weather model server, the weather model
        # name, time and bounding box may be needed to retrieve the dataset; for cases where no file is actually
        # downloaded, e.g. the GMAO and MERRA-2 models using OpenDAP, this function can be omitted leaving the data
        # retrieval to the following "load_weather" function.
        self._files = self._download_abcd_file(out, 'abcd', time, self._bounds)

    def load_weather(self, filename):
        '''
        Load weather model variables from the downloaded file named filename
        Inputs: 
        filename - filename of the downloaded weather model file 
        '''

        # Auxilliary function:
        # read individual variables (in 3-D cube format with exactly the same dimension) from downloaded file
        # This function needs to be writen by the users. For downloaded file from the weather model server,
        # this function extracts the individual variables from the saved file named filename;
        # for cases where no file is actually downloaded, e.g. the GMAO and MERRA-2 models using OpenDAP,
        # this function retrieves the individual variables directly from the weblink of the weather model.
        lats, lons, xs, ys, t, q, p, hgt = self._makeDataCubes(filename)

        # extra steps that may be needed to calculate topographic height and pressure level if not provided
        # directly by the weather model through the above auxilliary function "self._makeDataCubes"

        # if surface pressure (in logarithm) is provided as "p" along with the surface geopotential "z" (needs to be
        # added to the auxilliary function "self._makeDataCubes"), one can use the following line to convert to
        # geopotential, pressure level and geopotential height; otherwise commented out
        z, p, hgt = self._calculategeoh(z, p)

        # if the geopotential is provided as "z" (needs to be added to the auxilliary function "self._makeDataCubes"),
        # one can use the following line to convert to geopotential height; otherwise, commented out
        hgt = z / self._g0

        # if geopotential height is provided/calculated as "hgt", one can use the following line to convert to
        # topographic height, which is then automatically assigned to "self._zs"; otherwise commented out
        self._get_heights(lats, hgt)

        # if topographic height is provided as "hgt", use the following line directly; otherwise commented out
        self._zs = hgt

        ###########

        ######### output of the weather model reader for delay calculations (all in 3-D data cubes) ########

        # _t: temperture
        # _q: either relative or specific humidity
        # _p: must be pressure level
        # _xs: x-direction grid dimension of the native weather model coordinates (if in lat/lon, _xs = _lons)
        # _ys: y-direction grid dimension of the native weather model coordinates (if in lat/lon, _ys = _lats)
        # _zs: must be topographic height
        # _lats: latitude
        # _lons: longitude
        self._t = t
        self._q = q
        self._p = p
        self._xs = xs
        self._ys = ys
        self._lats = lats
        self._lons = lons

        ###########

    def _download_abcd_file(self, out, model_name, date_time, bounding_box):
        '''
        Auxilliary function:
        Download weather model data from a server
        Inputs: 
        out - filename for saving the retrieved data file 
        model_name - name of the custom weather model 
        date_time - datatime object (year,month,day,hour,minute,second)
        bounding_box - lat/lon bounding box for the region of interest
        Output: 
        out - returned filename from input
        '''
        pass

    def _makeDataCubes(self, filename):
        '''
        Auxilliary function:
        Read 3-D data cubes from downloaded file or directly from weather model weblink (in which case, there is no 
        need to download and save any file; rather, the weblink needs to be hardcoded in the custom reader, e.g. GMAO)
        Input: 
        filename - filename of the downloaded weather model file from the server
        Outputs: 
        lats - latitude (3-D data cube)
        lons - longitude (3-D data cube)
        xs - x-direction grid dimension of the native weather model coordinates (3-D data cube; if in lat/lon, _xs = _lons)
        ys - y-direction grid dimension of the native weather model coordinates (3-D data cube; if in lat/lon, _ys = _lats)
        t - temperature (3-D data cube)
        q - humidity (3-D data cube; could be relative humidity or specific humidity)
        p - pressure level (3-D data cube; could be pressure level (preferred) or surface pressure)
        hgt - height (3-D data cube; could be geopotential height or topographic height (preferred))
        '''
        pass
=======
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

