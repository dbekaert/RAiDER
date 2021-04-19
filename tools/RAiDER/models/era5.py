import datetime

import numpy as np
from pyproj import CRS

from RAiDER.models.ecmwf import ECMWF
from RAiDER.logger import *
from RAiDER.models.model_levels import A_137_ERA5, B_137_ERA5, LEVELS_137_HEIGHTS



class ERA5(ECMWF):
    # I took this from
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels.
    def __init__(self):
        ECMWF.__init__(self)

        self._humidityType = 'q'
        # Default, pressure levels are 'pl'
        self._model_level_type = 'pl'
        self._expver = '0001'
        self._classname = 'ea'
        self._dataset = 'era5'
        self._Name = 'ERA-5'
        self._proj = CRS.from_epsg(4326)

        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(1950, 1, 1), "Present")
        # Availability lag time in days
        self._lag_time = datetime.timedelta(days=30)

        self._a = A_137_ERA5
        self._b = B_137_ERA5

    def _fetch(self, lats, lons, time, out, Nextra=2):
        '''
        Fetch a weather model from ECMWF
        '''
        # bounding box plus a buffer
        lat_min, lat_max, lon_min, lon_max = self._get_ll_bounds(lats, lons, Nextra)

        # execute the search at ECMWF
        try:
            self._get_from_cds(
                lat_min, lat_max, self._lat_res, lon_min, lon_max, self._lon_res, time,
                out)
        except Exception as e:
            logger.warning(e)
            raise RuntimeError('Could not access or download from the CDS API')

    def load_weather(self, *args, **kwargs):
        self._load_pressure_level(*self.files, *args, **kwargs)

    def _load_pressure_level(self, filename, *args, **kwargs):
        import xarray as xr
        with xr.open_dataset(filename) as block:
            # Pull the data
            z = np.squeeze(block['z'].values)
            t = np.squeeze(block['t'].values)
            q = np.squeeze(block['q'].values)
            r = np.squeeze(block['r'].values)
            lats = np.squeeze(block.latitude.values)
            lons = np.squeeze(block.longitude.values)
            levels = np.squeeze(block.level.values) * 100

        z = np.flip(z, axis=1)

        # ECMWF appears to give me this backwards
        if lats[0] > lats[1]:
            z = z[::-1]
            t = t[:, ::-1]
            q = q[:, ::-1]
            r = r[:, ::-1]
            lats = lats[::-1]
        # Lons is usually ok, but we'll throw in a check to be safe
        if lons[0] > lons[1]:
            z = z[..., ::-1]
            t = t[..., ::-1]
            q = q[..., ::-1]
            r = r[..., ::-1]
            lons = lons[::-1]
        # pyproj gets fussy if the latitude is wrong, plus our
        # interpolator isn't clever enough to pick up on the fact that
        # they are the same
        lons[lons > 180] -= 360

        self._t = t
        self._q = q
        self._rh = r

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
        self._rh = np.transpose(self._rh)
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
        self._rh = self._rh.swapaxes(0, 1)
        self._p = self._p.swapaxes(0, 1)
        self._q = self._q.swapaxes(0, 1)
        self._t = self._t.swapaxes(0, 1)

        # For some reason z is opposite the others
        self._p = np.flip(self._p, axis=2)
        self._t = np.flip(self._t, axis=2)
        self._q = np.flip(self._q, axis=2)
        self._rh = np.flip(self._rh, axis=2)
