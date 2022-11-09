import datetime
import logging
import os
import shutil
import requests
import xarray

import numpy as np

from herbie import Herbie
from pathlib import Path
from pyproj import CRS

from RAiDER.logger import logger
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)


class HRRR(WeatherModel):
    def __init__(self):
        # initialize a weather model
        WeatherModel.__init__(self)

        self._humidityType = 'q'
        self._model_level_type = 'pl'  # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'hrrr'
        self._dataset = 'hrrr'

        # Tuple of min/max years where data is available.
        self._valid_range = (datetime.datetime(2016, 7, 15), "Present")
        self._lag_time = datetime.timedelta(hours=3)  # Availability lag time in days

        # model constants: TODO: need to update/double-check these
        self._k1 = 0.776  # [K/Pa]
        self._k2 = 0.233  # [K/Pa]
        self._k3 = 3.75e3  # [K^2/Pa]

        # 3 km horizontal grid spacing
        self._lat_res = 3. / 111
        self._lon_res = 3. / 111
        self._x_res = 3.
        self._y_res = 3.

        self._Nproc = 1
        self._Name = 'HRRR'
        self._Npl = 0
        self.files = None
        self._bounds = None
        self._zlevels = np.flipud(LEVELS_137_HEIGHTS)

        # Projection
        # See https://github.com/blaylockbk/pyBKB_v2/blob/master/demos/HRRR_earthRelative_vs_gridRelative_winds.ipynb and code lower down
        # '262.5:38.5:38.5:38.5 237.280472:1799:3000.00 21.138123:1059:3000.00'
        # 'lov:latin1:latin2:latd lon1:nx:dx lat1:ny:dy'
        # LCC parameters
        lon0 = 262.5
        lat0 = 38.5
        lat1 = 38.5
        lat2 = 38.5
        x0 = 0
        y0 = 0
        earth_radius = 6371229
        p1 = CRS('+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} +x_0={x0} +y_0={y0} +a={a} +b={a} +units=m +no_defs'.format(lat1=lat1, lat2=lat2, lat0=lat0, lon0=lon0, x0=x0, y0=y0, a=earth_radius))
        self._proj = p1

    def _fetch(self,  out):
        '''
        Fetch weather model data from HRRR
        '''
        # bounding box plus a buffer
        time = self._time

        self.files = self._download_hrrr_file(time, out)
        


    def load_weather(self, *args, filename=None, **kwargs):
        '''
        Load a weather model into a python weatherModel object, from self.files if no
        filename is passed.
        '''
        if filename is None:
            filename = self.files

        # read data from grib file
        pl = self._getPresLevels()
        pl = np.array([self._convertmb2Pa(p) for p in pl['Values']])
        ds = xarray.open_dataset(filename)
        xArr = ds['x'].values
        yArr = ds['y'].values
        lats = ds['lats'].values
        lons = ds['lons'].values
        temps = ds['t'].values
        qs = ds['q'].values
        geo_hgt = ds['z'].values

        Ny, Nx = lats.shape

        lons[lons > 180] -= 360

        # data cube format should be lats,lons,heights
        _xs = np.broadcast_to(xArr[np.newaxis, :, np.newaxis],
                              geo_hgt.shape)
        _ys = np.broadcast_to(yArr[:, np.newaxis, np.newaxis],
                              geo_hgt.shape)
        _lons = np.broadcast_to(lons[..., np.newaxis],
                                geo_hgt.shape)
        _lats = np.broadcast_to(lats[..., np.newaxis],
                                geo_hgt.shape)

        # correct for latitude
        self._get_heights(_lats, geo_hgt)

        self._t = temps
        self._q = qs
        self._p = np.broadcast_to(pl[np.newaxis, np.newaxis, :],
                                  self._zs.shape)
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons

        # For some reason z is opposite the others
        self._p = np.flip(self._p, axis=2)


    def _makeDataCubes(self, filename, out=None):
        '''
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels
        '''
        if out is None:
            out = filename

        # Pull the native grid
        xArr, yArr = self.getXY_gdal(filename)

        # open the dataset and pull the data
        ds = xarray.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'})
        t = ds['t'].values.copy()
        z = ds['gh'].values.copy()
        q = ds['q'].values.copy()
        lats = ds['t'].latitude.values.copy()
        lons = ds['t'].longitude.values.copy()

        ds_new = xarray.Dataset(
            data_vars=dict(
                t= (["y", "x", "level"], np.moveaxis(t, [0, 1, 2], [2, 0, 1])),
                z= (["y", "x", "level"], np.moveaxis(z, [0, 1, 2], [2, 0, 1])),
                q= (["y", "x", "level"], np.moveaxis(q, [0, 1, 2], [2, 0, 1])),
                lons=(["y", "x"], lons),
                lats=(["y", "x"], lats),
            ),
            coords=dict(
                level=np.arange(40) + 1, #TODO: is this correct? was 137...
                x=(["x"], xArr),
                y=(["y"], yArr),
            ),
            attrs={
                'Weather_model':'HRRR',
           }
        )
        ds_new.to_netcdf(out)


    def _getPresLevels(self, low=50, high=1013.2, inc=25):
        presList = [float(v) for v in range(int(low // 1), int(high // 1), int(inc // 1))]
        presList.append(high)
        outDict = {'Values': presList, 'units': 'mb', 'Name': 'Pressure_levels'}
        return outDict

    def _download_hrrr_file(self, DATE, out, model='hrrr', product='prs', fxx=0, verbose=False):
        '''
        Download a HRRR model
        '''
        H = Herbie(
            DATE.strftime('%Y-%m-%d %H:%M'),
            model=model,
            product=product,
            overwrite=False,
            verbose=True,
            save_dir=Path(os.path.dirname(out)),
        )
        pf = H.download(":(SPFH|PRES|TMP|HGT):", verbose=verbose)

        self._makeDataCubes(pf, out)

        return out