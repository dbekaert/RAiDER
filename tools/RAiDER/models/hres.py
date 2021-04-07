import datetime

import numpy as np
import xarray as xr

from pyproj import CRS

from RAiDER.utilFcns import round_date, convertLons
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
    LEVELS_91_HEIGHTS,
    A_137_HRES,
    B_137_HRES,
    A_91_HRES,
    B_91_HRES,
)


class HRES(WeatherModel):
    '''
    Implement ECMWF models
    '''

    def __init__(self, level_type='ml'):
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 9 km horizontal grid spacing. This is only used for extending the download-buffer, i.e. not in subsequent processing.
        self._lon_res = 9./111 #0.08108115
        self._lat_res = 9./111 #0.08108115
        self._x_res = 9./111 #0.08108115
        self._y_res = 9./111 #0.08108115

        self._humidityType = 'q'
        # Default, pressure levels are 'pl'
        self._level_type = 'ml'
        self._expver = '1'
        self._classname = 'od'
        self._dataset = 'hres'
        self._Name = 'HRES'
        self._proj = CRS.from_epsg(4326)

        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(1983, 4, 20), "Present")
        # Availability lag time in days
        self._lag_time = datetime.timedelta(hours=6)

        self.setLevel(level_type)

    
    def setLevel(self, levelType='ml'):
        '''Set the level type to model levels or pressure levels'''
        if levelType in ['ml', 'pl']:
            self._level_type = levelType
        else:
            raise RuntimeError('Level type {} is not recognized'.format(levelType))

        if levelType == 'ml':
            self.__model_levels__()
        else:
            self.__pressure_levels__()

    def __pressure_levels__(self):
        pass

    def __model_levels__(self):
        self._levels = 137
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)
        self._a = A_137_HRES
        self._b = B_137_HRES 

    def update_a_b(self):
        # Before 2013-06-26, there were only 91 model levels. The mapping coefficients below are extracted 
        # based on https://www.ecmwf.int/en/forecasts/documentation-and-support/91-model-levels
        self._levels = 91
        self._zlevels = np.flipud(LEVELS_91_HEIGHTS)
        self._a = A_91_HRES
        self._b = B_91_HRES


    def load_weather(self, filename=None):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        if filename is None:
            filename = self.files[0]

        if self._level_type == 'ml':
            self._load_model_levels(filename)
        elif self._level_type == 'pl':
            self._load_pressure_levels(filename)
       

    def _load_model_levels(self, filename):
        # read data from netcdf file
        lats, lons, xs, ys, t, q, lnsp, z = self._makeDataCubes(
            filename,
            verbose=False
        )

        if (self._time < datetime.datetime(2013, 6, 26, 0, 0, 0)):
            self.update_a_b()

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            lnsp = lnsp[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            lnsp = lnsp[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q
        geo_hgt, pres, hgt = self._calculategeoh(z, lnsp)

        # re-assign lons, lats to match heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :], hgt.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis], hgt.shape)
        # ys is latitude
        self._get_heights(_lats, hgt)
        h = self._zs.copy()

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            p = np.broadcast_to(pres[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        p = np.transpose(p)
        t = np.transpose(t)
        q = np.transpose(q)
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

        # Flip all the axis so that zs are in order from bottom to top
        p = np.flip(p, axis=2)
        t = np.flip(t, axis=2)
        q = np.flip(q, axis=2)
        h = np.flip(h, axis=2)
        _lats = np.flip(_lats, axis=2)
        _lons = np.flip(_lons, axis=2)

        self._p = p
        self._q = q
        self._t = t
        self._lats = _lats
        self._lons = _lons
        self._xs = _lons.copy()
        self._ys = _lats.copy()
        self._zs = h

    def _makeDataCubes(self, fname, verbose=False):
        '''
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels
        '''
        # get ll_bounds
        S, N, W, E = self._ll_bounds

        with xr.open_dataset(fname) as ds:
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180))

            # mask based on query bounds
            m1 = (S <= ds.latitude) & (N >= ds.latitude)
            m2 = (W <= ds.longitude) & (E >= ds.longitude)
            block = ds.where(m1 & m2, drop=True)

            # Pull the data
            z = np.squeeze(block['z'].values)[0, ...]
            t = np.squeeze(block['t'].values)
            q = np.squeeze(block['q'].values)
            lnsp = np.squeeze(block['lnsp'].values)[0, ...]
            lats = np.squeeze(block.latitude.values)
            lons = np.squeeze(block.longitude.values)

            xs = lons.copy()
            ys = lats.copy()

        if z.size == 0:
            raise RuntimeError('There is no data in z, '
                               'you may have a problem with your mask')

        return lats, lons, xs, ys, t, q, lnsp, z

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        if (time < datetime.datetime(2013, 6, 26, 0, 0, 0)):
            self.update_a_b()

        # execute the search at ECMWF
        self._download_ecmwf(lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time, out)

    def _download_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out):
        from ecmwfapi import ECMWFService

        server = ECMWFService("mars")

        corrected_date = round_date(time, datetime.timedelta(hours=6))

        if self._model_level_type == 'ml':
            param = "129/130/133/152"
        else:
            param = "129.128/130.128/133.128/152"

        server.execute(
            {
                'class': self._classname,
                'dataset': self._dataset,
                'expver': "{}".format(self._expver),
                'resol': "av",
                'stream': "oper",
                'type': "an",
                'levelist': "all",
                'levtype': "{}".format(self._model_level_type),
                'param': param,
                'date': datetime.datetime.strftime(corrected_date, "%Y-%m-%d"),
                'time': "{}".format(datetime.time.strftime(corrected_date.time(), '%H:%M')),
                'step': "0",
                'grid': "{}/{}".format(lon_step, lat_step),
                'area': "{}/{}/{}/{}".format(lat_max, floorish(lon_min, 0.1), floorish(lat_min, 0.1), lon_max),
                'format': "netcdf",
            },
            out
        )


    def _load_pressure_levels(self, filename, *args, **kwargs):
        import xarray as xr
        with xr.open_dataset(filename) as block:
            # Pull the data
            z = np.squeeze(block['z'].values)
            t = np.squeeze(block['t'].values)
            q = np.squeeze(block['q'].values)
            lats = np.squeeze(block.latitude.values)
            lons = np.squeeze(block.longitude.values)
            levels = np.squeeze(block.level.values) * 100

        z = np.flip(z, axis=1)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q

        geo_hgt = z / self._g0

        # re-assign lons, lats to match heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],
                                geo_hgt.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],
                                geo_hgt.shape)

        # correct heights for latitude
        self._get_heights(_lats, geo_hgt)

        self._p = np.broadcast_to(levels[:, np.newaxis, np.newaxis],
                                  self._zs.shape)

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p)
        self._t = np.transpose(self._t)
        self._q = np.transpose(self._q)
        self._lats = np.transpose(_lats)
        self._lons = np.transpose(_lons)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()
        self._zs = np.transpose(self._zs)

        # check this
        # data cube format should be lats,lons,heights
        self._lats = self._lats.swapaxes(0, 1)
        self._lons = self._lons.swapaxes(0, 1)
        self._xs = self._xs.swapaxes(0, 1)
        self._ys = self._ys.swapaxes(0, 1)
        self._zs = self._zs.swapaxes(0, 1)
        self._p = self._p.swapaxes(0, 1)
        self._q = self._q.swapaxes(0, 1)
        self._t = self._t.swapaxes(0, 1)

        # For some reason z is opposite the others
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)

def floorish(val, frac):
    '''Round a value to the lower fractional part'''
    return val - (val % frac)

