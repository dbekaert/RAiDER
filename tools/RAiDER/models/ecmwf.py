import datetime as dt

import numpy as np
import xarray as xr
from pyproj import CRS

from RAiDER import utilFcns as util
from RAiDER.logger import logger
from RAiDER.models.model_levels import (
    A_137_HRES,
    B_137_HRES,
    LEVELS_25_HEIGHTS,
    LEVELS_137_HEIGHTS,
)
from RAiDER.models.weatherModel import TIME_RES, WeatherModel


class ECMWF(WeatherModel):
    """Implement ECMWF models."""

    def __init__(self) -> None:
        # initialize a weather model
        WeatherModel.__init__(self)

        # model constants
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        self._time_res = TIME_RES['ECMWF']

        self._lon_res = 0.25
        self._lat_res = 0.25
        self._proj = CRS.from_epsg(4326)

        self._model_level_type = 'ml'  # Default

    def __pressure_levels__(self):
        self._zlevels = np.flipud(LEVELS_25_HEIGHTS)
        self._levels = len(self._zlevels)

    def __model_levels__(self):
        self._levels = 137
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)
        self._a = A_137_HRES
        self._b = B_137_HRES

    def load_weather(self, f=None, *args, **kwargs) -> None:
        """
        Consistent class method to be implemented across all weather model types.
        As a result of calling this method, all of the variables (x, y, z, p, q,
        t, wet_refractivity, hydrostatic refractivity, e) should be fully
        populated.
        """
        f = f if f is not None else self.files[0]
        self._load_model_level(f)

    def _load_model_level(self, fname) -> None:
        # read data from netcdf file
        lats, lons, xs, ys, t, q, lnsp, z = self._makeDataCubes(fname, verbose=False)

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

        self._lons, self._lats = np.meshgrid(lons, lats)

        # ys is latitude
        self._get_heights(self._lats, hgt.transpose(1, 2, 0))
        h = self._zs.copy()

        # We want to support both pressure levels and true pressure grids.
        # If the shape has one dimension, we'll scale it up to act as a
        # grid, otherwise we'll leave it alone.
        if len(pres.shape) == 1:
            self._p = np.broadcast_to(pres[:, np.newaxis, np.newaxis], self._zs.shape)
        else:
            self._p = pres

        # Re-structure everything from (heights, lats, lons) to (lons, lats, heights)
        self._p = self._p.transpose(1, 2, 0)
        self._t = self._t.transpose(1, 2, 0)
        self._q = self._q.transpose(1, 2, 0)

        # Flip all the axis so that zs are in order from bottom to top
        # lats / lons are simply replicated to all heights so they don't need flipped
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()
        self._zs = np.flip(h, axis=2)

    def _fetch(self, out) -> None:
        """Fetch a weather model from ECMWF."""
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._ll_bounds
        # execute the search at ECMWF
        self._get_from_ecmwf(lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, self._time, out)

    def _get_from_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out) -> None:
        import ecmwfapi

        server = ecmwfapi.ECMWFDataServer()

        corrected_DT = util.round_date(time, dt.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

        server.retrieve(
            {
                'class': self._classname,  # ERA-Interim
                'dataset': self._dataset,
                'expver': f'{self._expver}',
                # They warn me against all, but it works well
                'levelist': 'all',
                'levtype': 'ml',  # Model levels
                'param': 'lnsp/q/z/t',  # Necessary variables
                'stream': 'oper',
                # date: Specify a single date as "2015-08-01" or a period as
                # "2015-08-01/to/2015-08-31".
                'date': dt.datetime.strftime(corrected_DT, '%Y-%m-%d'),
                # type: Use an (analysis) unless you have a particular reason to
                # use fc (forecast).
                'type': 'an',
                # time: With type=an, time can be any of
                # "00:00:00/06:00:00/12:00:00/18:00:00".  With type=fc, time can
                # be any of "00:00:00/12:00:00",
                'time': dt.time.strftime(corrected_DT.time(), '%H:%M:%S'),
                # step: With type=an, step is always "0". With type=fc, step can
                # be any of "3/6/9/12".
                'step': '0',
                # grid: Only regular lat/lon grids are supported.
                'grid': f'{lat_step}/{lon_step}',
                'area': f'{lat_max}/{lon_min}/{lat_min}/{lon_max}',  # area: N/W/S/E
                'format': 'netcdf',
                'resol': 'av',
                'target': out,  # target: the name of the output file.
            }
        )

    def _get_from_cds(self, lat_min, lat_max, lon_min, lon_max, acqTime, outname) -> None:
        """Used for ERA5."""
        import cdsapi

        c = cdsapi.Client(verify=0)

        if self._model_level_type == 'pl':
            var = ['z', 'q', 't']
        else:
            var = '129/130/133/152'  # 'lnsp', 'q', 'z', 't'

        bbox = [lat_max, lon_min, lat_min, lon_max]

        # round to the closest legal time

        corrected_DT = util.round_date(acqTime, dt.timedelta(hours=self._time_res))
        if not corrected_DT == acqTime:
            logger.warning('Rounded given datetime from  %s to %s', acqTime, corrected_DT)

        # I referenced https://confluence.ecmwf.int/display/CKB/How+to+download+ERA5
        dataDict = {
            'class': 'ea',
            'expver': '1',
            'levelist': 'all',
            'levtype': f'{self._model_level_type}',  # 'ml' for model levels or 'pl' for pressure levels
            'param': var,
            'stream': 'oper',
            'type': 'an',
            'date': corrected_DT.strftime('%Y-%m-%d'),
            'time': dt.time.strftime(corrected_DT.time(), '%H:%M'),
            # step: With type=an, step is always "0". With type=fc, step can
            # be any of "3/6/9/12".
            'step': '0',
            'area': bbox,
            'grid': [0.25, 0.25],
            'format': 'netcdf',
        }

        try:
            c.retrieve('reanalysis-era5-complete', dataDict, outname)
        except:
            raise Exception

    def _download_ecmwf(self, lat_min, lat_max, lat_step, lon_min, lon_max, lon_step, time, out) -> None:
        """Used for HRES."""
        from ecmwfapi import ECMWFService

        server = ECMWFService('mars')

        # round to the closest legal time
        corrected_DT = util.round_date(time, dt.timedelta(hours=self._time_res))
        if not corrected_DT == time:
            logger.warning('Rounded given datetime from  %s to %s', time, corrected_DT)

        if self._model_level_type == 'ml':
            param = '129/130/133/152'
        else:
            param = '129.128/130.128/133.128/152'

        server.execute(
            {
                'class': self._classname,
                'dataset': self._dataset,
                'expver': f'{self._expver}',
                'resol': 'av',
                'stream': 'oper',
                'type': 'an',
                'levelist': 'all',
                'levtype': f'{self._model_level_type}',
                'param': param,
                'date': dt.datetime.strftime(corrected_DT, '%Y-%m-%d'),
                'time': dt.time.strftime(corrected_DT.time(), '%H:%M'),
                'step': '0',
                'grid': f'{lon_step}/{lat_step}',
                'area': f'{lat_max}/{util.floorish(lon_min, 0.1)}/{util.floorish(lat_min, 0.1)}/{lon_max}',
                'format': 'netcdf',
            },
            out,
        )

    def _load_pressure_level(self, filename, *args, **kwargs) -> None:
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

        geo_hgt = (z / self._g0).transpose(1, 2, 0)

        # re-assign lons, lats to match heights
        self._lons, self._lats = np.meshgrid(lons, lats)

        # correct heights for latitude
        self._get_heights(self._lats, geo_hgt)

        self._p = np.broadcast_to(levels[np.newaxis, np.newaxis, :], self._zs.shape)

        # Re-structure from (heights, lats, lons) to (lons, lats, heights)
        self._t = self._t.transpose(1, 2, 0)
        self._q = self._q.transpose(1, 2, 0)
        self._ys = self._lats.copy()
        self._xs = self._lons.copy()

        # flip z to go from surface to toa
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)

    def _makeDataCubes(self, fname, verbose=False):
        """
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels.
        """
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
            raise RuntimeError('There is no data in z, you may have a problem with your mask')

        return lats, lons, xs, ys, t, q, lnsp, z
