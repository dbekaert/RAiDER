"""
Created on Wed Sep  9 10:26:44 2020 @author: prashant
Modified by Yang Lei, GPS/Caltech
"""
import datetime
import os
import urllib.request

import numpy as np

from pyproj import CRS

from RAiDER.models.weatherModel import WeatherModel, TIME_RES
from RAiDER.logger import logger
from RAiDER.utilFcns import (
    writeWeatherVars2NETCDF4,
    read_NCMR_loginInfo,
    show_progress
)
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)


class NCMR(WeatherModel):
    '''
    Implement NCMRWF NCUM (named as NCMR) model in future
    '''

    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'                     # q for specific humidity and rh for relative humidity
        self._model_level_type = 'ml'                # Default, pressure levels are 'pl', and model levels are "ml"
        self._classname = 'ncmr'                     # name of the custom weather model
        self._dataset = 'ncmr'                       # same name as above
        self._Name = 'NCMR'                          # name of the new weather model (in Capital)
        self._time_res = TIME_RES[self._dataset.upper()]

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

        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)

        self._bounds = None

        # Projection
        self._proj = CRS.from_epsg(4326)

    def _fetch(self, out):
        '''
        Fetch weather model data from NCMR: note we only extract the lat/lon bounds for this weather model;
        fetching data is not needed here as we don't actually download data , data exist in same system
        '''
        time = self._time

        # Auxillary function:
        '''
        download data of the NCMR model and save it in desired location
        '''
        self._files = self._download_ncmr_file(out, time, self._ll_bounds)

    def load_weather(self, filename=None, *args, **kwargs):
        '''
        Load NCMR model variables from existing file
        '''
        if filename is None:
            filename = self.files[0]

        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._ll_bounds
        self._bounds = (lat_min, lat_max, lon_min, lon_max)

        self._makeDataCubes(filename)

    def _download_ncmr_file(self, out, date_time, bounding_box):
        '''
        Download weather model data (whole globe) from NCMR weblink, crop it to the region of interest, and save the cropped data as a standard .nc file of RAiDER (e.g. "NCMR_YYYY_MM_DD_THH_MM_SS.nc");
        Temporarily download data from NCMR ftp 'https://ftp.ncmrwf.gov.in/pub/outgoing/SAC/NCUM_OSF/' and copied in weather_models folder
        '''

        from netCDF4 import Dataset

        ############# Use these lines and modify the link when actually downloading NCMR data from a weblink #############
        url, username, password = read_NCMR_loginInfo()
        filename = os.path.basename(out)
        url = f'ftp://{username}:{password}@{url}/TEST/{filename}'
        filepath = f'{out[:-3]}_raw.nc'
        if not os.path.exists(filepath):
            logger.info('Fetching URL: %s', url)
            local_filename, headers = urllib.request.urlretrieve(url, filepath, show_progress)
        else:
            logger.warning('Weather model already exists, skipping download')
        ########################################################################################################################

        ############# For debugging: use pre-downloaded files; Remove/comment out it when actually downloading NCMR data from a weblink #############
#        filepath = os.path.dirname(out) + '/NCUM_ana_mdllev_20180701_00z.nc'
        ########################################################################################################################

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

        with Dataset(filepath, 'r', maskandscale=True) as f:
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

        ############# For debugging: comment it out when using pre-downloaded raw data files and don't want to remove them for test; Uncomment it when actually downloading NCMR data from a weblink #############
        os.remove(filepath)
        ########################################################################################################################

        try:
            writeWeatherVars2NETCDF4(self, lats, lons, hgt, q, p, t, outName=out)
        except Exception:
            logger.exception("Unable to save weathermodel to file")

    def _makeDataCubes(self, filename):
        '''
        Get the variables from the saved .nc file (named as "NCMR_YYYY_MM_DD_THH_MM_SS.nc")
        '''
        from netCDF4 import Dataset

        # adding the import here should become absolute when transition to netcdf
        with Dataset(filename, mode='r') as f:
            lons = np.array(f.variables['x'][:])
            lats = np.array(f.variables['y'][:])
            hgt = np.array(f.variables['H'][:])
            q = np.array(f.variables['QV'][:])
            p = np.array(f.variables['PL'][:])
            t = np.array(f.variables['T'][:])

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
