"""
Created on Wed Sep  9 10:26:44 2020 @author: prashant
Modified by Yang Lei, GPS/Caltech
"""
import datetime
import numpy as np
from pyproj import CRS
from RAiDER import utilFcns as util
from RAiDER.models.weatherModel import WeatherModel


class NCMR(WeatherModel):
    '''
    Implement NCMRWF NCUM (named as NCMR) model in future
    '''

    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)
        self._model_file_type = 'nc'

        self._humidityType = 'q'                     # q for specific humidity and rh for relative humidity
        self._model_level_type = 'ml'                # Default, pressure levels are 'pl', and model levels are "ml"
        self._classname = 'ncmr'                     # name of the custom weather model
        self._dataset = 'ncmr'                       # same name as above
        self._Name = 'NCMR'                          # name of the new weather model (in Capital)

        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(2015, 12, 1), "Present")
        # Availability lag time in days/hours
        self._lag_time = datetime.timedelta(hours=6)

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # horizontal grid spacing
        self._lon_res = .17578125                  # grid spacing in longitude
        self._lat_res = .11718750                  # grid spacing in latitude

        self._x_res = .17578125                  # same as longitude
        self._y_res = .11718750                  # same as latitude

        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch weather model data from NCMR: note we only extract the lat/lon bounds for this weather model;
        fetching data is not needed here as we don't actually download data , data exist in same system
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        # Auxillary function:
        '''
        download data of the NCMR model and save it in desired location
        '''
        self._files = self._download_ncmr_file(out, 'ncmr', time, self._bounds)

    def load_weather(self, filename):
        '''
        Load NCMR model variables from existing file
        '''
        self._makeDataCubes(filename)

    def _download_ncmr_file(self, out, ncmr, date_time, bounding_box):
        '''
        Temporarily download data from NCMR ftp 'https://ftp.ncmrwf.gov.in/pub/outgoing/SAC/NCUM_OSF/' and copied in weather_models folder
        '''
        pass

    def _makeDataCubes(self, filename):

        from netCDF4 import Dataset

        # calculate the array indices for slicing the GMAO variable arrays
        lat_min_ind = int((self._bounds[0] - (-89.94141)) / self._lat_res)
        lat_max_ind = int((self._bounds[1] - (-89.94141)) / self._lat_res)
        if (self._bounds[2] < 0.0):
            lon_min_ind = int((self._bounds[2] + 360.0 - (0.087890625)) / self._lon_res)
        else:
            lon_min_ind = int((self._bounds[2] - (0.087890625)) / self._lon_res)
        if (self._bounds[3] < 0.0):
            lon_max_ind = int((self._bounds[3] + 360.0 - (0.087890625)) / self._lon_res)
        else:
            lon_max_ind = int((self._bounds[3] - (0.087890625)) / self._lon_res)

        ml_min = 0
        ml_max = 70

        with Dataset(filename, 'r', maskandscale=True) as f:
            lats = f.variables['latitude'][lat_min_ind:(lat_max_ind + 1)].copy()
            if (self._bounds[2] * self._bounds[3] < 0):
                lons1 = f.variables['longitude'][lon_min_ind:].copy()
                lons2 = f.variables['longitude'][0:(lon_max_ind + 1)].copy()
                lons = np.append(lons1, lons2)
            else:
                lons = f.variables['longitude'][lon_min_ind:(lon_max_ind + 1)].copy()
            if (self._bounds[2] * self._bounds[3] < 0):
                t1 = f.variables['air_temperature'][ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:].copy()
                t2 = f.variables['air_temperature'][ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), 0:(lon_max_ind + 1)].copy()
                t = np.append(t1, t2, axis=2)
            else:
                t = f.variables['air_temperature'][ml_min:(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)].copy()

            # Skipping first pressure levels (below 20 meter)
            if (self._bounds[2] * self._bounds[3] < 0):
                q1 = f.variables['specific_humidity'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:].copy()
                q2 = f.variables['specific_humidity'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), 0:(lon_max_ind + 1)].copy()
                q = np.append(q1, q2, axis=2)
            else:
                q = f.variables['specific_humidity'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)].copy()
            if (self._bounds[2] * self._bounds[3] < 0):
                p1 = f.variables['air_pressure'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:].copy()
                p2 = f.variables['air_pressure'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), 0:(lon_max_ind + 1)].copy()
                p = np.append(p1, p2, axis=2)
            else:
                p = f.variables['air_pressure'][(ml_min + 1):(ml_max + 1), lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)].copy()

            level_hgt = f.variables['level_height'][(ml_min + 1):(ml_max + 1)].copy()
            if (self._bounds[2] * self._bounds[3] < 0):
                surface_alt1 = f.variables['surface_altitude'][lat_min_ind:(lat_max_ind + 1), lon_min_ind:].copy()
                surface_alt2 = f.variables['surface_altitude'][lat_min_ind:(lat_max_ind + 1), 0:(lon_max_ind + 1)].copy()
                surface_alt = np.append(surface_alt1, surface_alt2, axis=1)
            else:
                surface_alt = f.variables['surface_altitude'][lat_min_ind:(lat_max_ind + 1), lon_min_ind:(lon_max_ind + 1)].copy()

            hgt = np.zeros([len(level_hgt), len(surface_alt[:, 1]), len(surface_alt[1, :])])
            for i in range(len(level_hgt)):
                hgt[i, :, :] = surface_alt[:, :] + level_hgt[i]

            lons[lons > 180] -= 360

            # re-assign lons, lats to match heights
            _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],
                                    t.shape)
            _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],
                                    t.shape)

            # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
            _lats = np.transpose(_lats)
            _lons = np.transpose(_lons)
            t = np.transpose(t)
            q = np.transpose(q)
            p = np.transpose(p)
            hgt = np.transpose(hgt)
            del surface_alt, i, level_hgt

            # data cube format should be lats,lons,heights
            p = p.swapaxes(0, 1)
            q = q.swapaxes(0, 1)
            t = t.swapaxes(0, 1)
            hgt = hgt.swapaxes(0, 1)
            _lats = _lats.swapaxes(0, 1)
            _lons = _lons.swapaxes(0, 1)

            # assign the regular-grid variables
            self._p = p
            self._q = q
            self._t = t
            self._lats = _lats
            self._lons = _lons
            self._xs = _lons
            self._ys = _lats
            self._zs = hgt
