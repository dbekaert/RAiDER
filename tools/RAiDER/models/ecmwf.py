import datetime

import numpy as np
from pyproj import CRS

from RAiDER.logger import *
from RAiDER import utilFcns as util
from RAiDER.models.weatherModel import WeatherModel


class ECMWF(WeatherModel):
    '''
    Implement ECMWF models
    '''

    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._model_file_type = 'nc'

        # model constants
        self._k1 = 0.776   # [K/Pa]
        self._k2 = 0.233   # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._lon_res = 0.2
        self._lat_res = 0.2

    def load_weather(self, filename):
        '''
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        '''
        self._load_model_level(filename)

    def _load_model_level(self, fname):
        from scipy.io import netcdf as nc
        with nc.netcdf_file(fname, 'r', maskandscale=True) as f:
            # 0,0 to get first time and first level
            z = f.variables['z'][0][0].copy()
            lnsp = f.variables['lnsp'][0][0].copy()
            t = f.variables['t'][0].copy()
            qq = f.variables['q'][0].copy()
            lats = f.variables['latitude'][:].copy()
            lons = f.variables['longitude'][:].copy()
            self._levels = f.variables['level'][:].copy()

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            lnsp = lnsp[::-1]
            t = t[:, ::-1]
            Q = qq[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            lnsp = lnsp[..., ::-1]
            t = t[..., ::-1]
            Q = qq[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360
        self._proj = CRS.from_epsg(4326)

        self._t = t
        self._q = Q

        geo_hgt, pres, hgt = self._calculategeoh(z, lnsp)

        # re-assign lons, lats to match heights
        _lons = np.broadcast_to(lons[np.newaxis, np.newaxis, :],
                                hgt.shape)
        _lats = np.broadcast_to(lats[np.newaxis, :, np.newaxis],
                                hgt.shape)
        # ys is latitude
        self._get_heights(_lats, hgt)

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            self._p = np.broadcast_to(pres[:, np.newaxis, np.newaxis],
                                      self._zs.shape)
        else:
            self._p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = np.transpose(self._p)
        self._t = np.transpose(self._t)
        self._q = np.transpose(self._q)
        self._lats = np.transpose(_lats)
        self._lons = np.transpose(_lons)
        self._zs = np.transpose(self._zs)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()

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

        # Flip all the axis so that zs are in order from bottom to top
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)
        self._zs = np.flip(self._zs, axis=2)

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at ECMWF
        try:
            self._get_from_ecmwf(
                lat_min, 
                lat_max, 
                self._lat_res, 
                lon_min, 
                lon_max, 
                self._lon_res, 
                time,
                out
            )
        except Exception as e:
            logger.warning('Query point bounds are {}/{}/{}/{}'.format(lat_min, lat_max, lon_min, lon_max))
            logger.warning('Query time: {}'.format(time))
            logger.exception(e)

    def _get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max,
                        lon_step, time, out):
        import ecmwfapi

        server = ecmwfapi.ECMWFDataServer()

        corrected_date = util.round_date(time, datetime.timedelta(hours=6))

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
            "date": datetime.datetime.strftime(corrected_date, "%Y-%m-%d"),
            # type: Use an (analysis) unless you have a particular reason to
            # use fc (forecast).
            "type": "an",
            # time: With type=an, time can be any of
            # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
            # be any of "00:00:00/12:00:00",
            "time": datetime.time.strftime(corrected_date.time(), "%H:%M:%S"),
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

    def _get_from_cds(self, lat_min, lat_max, lat_step, lon_min, lon_max,
                      lon_step, acqTime, outname):
        import cdsapi

        pls = ['1', '2', '3', '5', '7', '10', '20', '30', '50', '70', '100', '125', '150', '175', '200', '225', '250', '300', '350', '400', '450', '500', '550', '600', '650', '700', '750', '775', '800', '825', '850', '875', '900', '925', '950', '975', '1000']
        mls = np.arange(137) + 1

        c = cdsapi.Client(verify=0)
        # corrected_date = util.round_date(time, datetime.timedelta(hours=6))
        if self._model_level_type == 'pl':
            var = ['geopotential', 'relative_humidity', 'specific_humidity', 'temperature']
            levels = 'all'
            levType = 'pressure_level'
        else:
            var = ['lnsp', 'q', 'z', 't']
            levels = mls
            levType = 'model_level'

        bbox = [lat_max, lon_min, lat_min, lon_max]

        dataDict = {
            "product_type": "reanalysis",
            "{}".format(levType): levels,
            "levtype": "{}".format(self._model_level_type),  # 'ml' for model levels or 'pl' for pressure levels
            'variable': var,
            "stream": "oper",
            "type": "an",
            "year": "{}".format(acqTime.year),
            "month": "{}".format(acqTime.month),
            "day": "{}".format(acqTime.day),
            "time": "{}".format(datetime.time.strftime(acqTime.time(), '%H:%M')),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            "step": "0",
            "area": bbox,
            "format": "netcdf"}

        try:
            c.retrieve('reanalysis-era5-pressure-levels', dataDict, outname)
        except Exception as e:
            logger.warning('Query point bounds are {}/{} latitude and {}/{} longitude'.format(lat_min, lat_max, lon_min, lon_max))
            logger.warning('Query time: {}'.format(acqTime))
            logger.exception(e)
            raise Exception
