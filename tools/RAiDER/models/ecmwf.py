from abc import abstractmethod
import datetime

import numpy as np
import xarray as xr

from pyproj import CRS

from RAiDER.logger import logger
from RAiDER import utilFcns as util
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
    LEVELS_25_HEIGHTS,
    A_137_HRES,
    B_137_HRES,
)

from RAiDER.models.weatherModel import WeatherModel, TIME_RES


class ECMWF(WeatherModel):
    '''
    Implement ECMWF models
    '''

    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._time_res = TIME_RES['ECMWF']

        self._lon_res = 0.2
        self._lat_res = 0.2
        self._proj = CRS.from_epsg(4326)

        self._model_level_type = 'ml'  # Default

    def setLevelType(self, levelType):
        '''Set the level type to model levels or pressure levels'''
        if levelType in ['ml', 'pl']:
            self._model_level_type = levelType
        else:
            raise RuntimeError('Level type {} is not recognized'.format(levelType))

        if levelType == 'ml':
            self.__model_levels__()
        else:
            self.__pressure_levels__()

    def __pressure_levels__(self):
        self._zlevels = np.flipud(LEVELS_25_HEIGHTS)
        self._levels = len(self._zlevels)

    def __model_levels__(self):
        self._levels = 137
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)
        self._a = A_137_HRES
        self._b = B_137_HRES

    def load_weather(self, *args, **kwargs):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        self._load_model_level(*self.files)

    def _load_model_level(self, fname):
        # read data from netcdf file
        lats, lons, xs, ys, t, q, lnsp, z = self._makeDataCubes(
            fname,
            verbose=False
        )

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
            self._p = np.broadcast_to(pres[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            self._p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p, (1, 2, 0))
        self._t = np.transpose(self._t, (1, 2, 0))
        self._q = np.transpose(self._q, (1, 2, 0))
        h = np.transpose(h, (1, 2, 0))
        self._lats = np.transpose(_lats, (1, 2, 0))
        self._lons = np.transpose(_lons, (1, 2, 0))

        # Flip all the axis so that zs are in order from bottom to top
        # lats / lons are simply replicated to all heights so they don't need flipped
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()
        self._zs = np.flip(h, axis=2)

    def _fetch(self, out):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._ll_bounds

        # execute the search at ECMWF
        self._get_from_ecmwf(
            lat_min,
            lat_max,
            self._lat_res,
            lon_min,
            lon_max,
            self._lon_res,
            self._time,
            out
        )
        return


    def _get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max,
                        lon_step, time, out):
        import ecmwfapi

        server = ecmwfapi.ECMWFDataServer()

        corrected_DT = util.round_date(time, datetime.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

        server.retrieve({
            "class": self._classname,  # ERA-Interim
            'dataset': self._dataset,
            "expver": "{}".format(self._expver),
            # They warn me against all, but it works well
            "levelist": 'all',
            "levtype": "ml",  # Model levels
            "param": "lnsp/q/z/t",  # Necessary variables
            "stream": "oper",
            # date: Specify a single date as "2015-08-01" or a period as
            # "2015-08-01/to/2015-08-31".
            "date": datetime.datetime.strftime(corrected_DT, "%Y-%m-%d"),
            # type: Use an (analysis) unless you have a particular reason to
            # use fc (forecast).
            "type": "an",
            # time: With type=an, time can be any of
            # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
            # be any of "00:00:00/12:00:00",
            "time": datetime.time.strftime(corrected_DT.time(), "%H:%M:%S"),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            "step": "0",
            # grid: Only regular lat/lon grids are supported.
            "grid": '{}/{}'.format(lat_step, lon_step),
            "area": '{}/{}/{}/{}'.format(lat_max, lon_min, lat_min, lon_max),  # area: N/W/S/E
            "format": "netcdf",
            "resol": "av",
            "target": out,    # target: the name of the output file.
        })

    def _get_from_cds(
        self,
        lat_min,
        lat_max,
        lon_min,
        lon_max,
        acqTime,
        outname
    ):
        """ Used for ERA5 """
        import cdsapi
        c = cdsapi.Client(verify=0)

        if self._model_level_type == 'pl':
            var = ['z', 'q', 't']
            levType = 'pressure_level'
        else:
            var = "129/130/133/152"  # 'lnsp', 'q', 'z', 't'
            levType = 'model_level'

        bbox = [lat_max, lon_min, lat_min, lon_max]

        # round to the closest legal time

        corrected_DT = util.round_date(acqTime, datetime.timedelta(hours=self._time_res))
        if not corrected_DT == acqTime:
            logger.warning('Rounded given datetime from  %s to %s', acqTime, corrected_DT)


        # I referenced https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
        dataDict = {
            "class": "ea",
            "expver": "1",
            "levelist": 'all',
            "levtype": "{}".format(self._model_level_type),  # 'ml' for model levels or 'pl' for pressure levels
            'param': var,
            "stream": "oper",
            "type": "an",
            "date": "{}".format(corrected_DT.strftime('%Y-%m-%d')),
            "time": "{}".format(datetime.time.strftime(corrected_DT.time(), '%H:%M')),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            "step": "0",
            "area": bbox,
            "grid": [0.25, .25],
            "format": "netcdf"}

        try:
            c.retrieve('reanalysis-era5-complete', dataDict, outname)
        except Exception as e:
            raise Exception


    def _download_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out):
        """ Used for HRES """
        from ecmwfapi import ECMWFService

        server = ECMWFService("mars")

        # round to the closest legal time
        corrected_DT = util.round_date(time, datetime.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

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
                'date': datetime.datetime.strftime(corrected_DT, "%Y-%m-%d"),
                'time': "{}".format(datetime.time.strftime(corrected_DT.time(), '%H:%M')),
                'step': "0",
                'grid': "{}/{}".format(lon_step, lat_step),
                'area': "{}/{}/{}/{}".format(lat_max, util.floorish(lon_min, 0.1), util.floorish(lat_min, 0.1), lon_max),
                'format': "netcdf",
            },
            out
        )

    def _load_pressure_level(self, filename, *args, **kwargs):
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
