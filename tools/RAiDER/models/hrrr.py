import datetime
import logging
import os
import shutil
import requests
import xarray

import numpy as np

from herbie import Herbie
from pathlib import Path
from pyproj import CRS, Transformer

from RAiDER.logger import logger
from RAiDER.utilFcns import rio_profile, rio_extents, round_date
from RAiDER.models.weatherModel import (
    WeatherModel, transform_coords, TIME_RES
)
from RAiDER.models.model_levels import (
    LEVELS_137_HEIGHTS,
)


class HRRR(WeatherModel):
    def __init__(self):
        # initialize a weather model
        super().__init__()

        self._humidityType = 'q'
        self._model_level_type = 'pl'  # Default, pressure levels are 'pl'
        self._expver = '0001'
        self._classname = 'hrrr'
        self._dataset = 'hrrr'

        self._time_res = TIME_RES[self._dataset.upper()]

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
        p1 = CRS(f'+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} '\
                 f'+lon_0={lon0} +x_0={x0} +y_0={y0} +a={earth_radius} '\
                 f'+b={earth_radius} +units=m +no_defs')
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
        try:
            ds = xarray.open_dataset(filename, engine='cfgrib')
        except EOFError:
            ds = xarray.open_dataset(filename, engine='netcdf4')

        pl = np.array([self._convertmb2Pa(p) for p in ds.levels.values])
        xArr = ds['x'].values
        yArr = ds['y'].values
        lats = ds['lats'].values
        lons = ds['lons'].values
        temps = ds['t'].values.transpose(1, 2, 0)
        qs = ds['q'].values.transpose(1, 2, 0)
        geo_hgt = ds['z'].values.transpose(1, 2, 0)

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
                                  geo_hgt.shape)
        self._xs = _xs
        self._ys = _ys
        self._lats = _lats
        self._lons = _lons


    def _makeDataCubes(self, filename, out=None):
        '''
        Create a cube of data representing temperature and relative humidity
        at specified pressure levels
        '''
        if out is None:
            out = filename

        # Get profile information from gdal
        prof = rio_profile(str(filename))

        # Now get bounds
        S, N, W, E = self._ll_bounds

        # open the dataset and pull the data
        ds = xarray.open_dataset(filename, engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa'})

        # Determine mask based on query bounds
        lats = ds["latitude"].to_numpy()
        lons = ds["longitude"].to_numpy()
        levels = ds["isobaricInhPa"].to_numpy()
        shp = lats.shape
        lons[lons > 180.0] -= 360.
        m1 = (S <= lats) & (N >= lats) &\
                (W <= lons) & (E >= lons)

        if np.sum(m1) == 0:
            raise RuntimeError('Area of Interest has no overlap with the HRRR model available extent')

        # Y extent
        m1_y = np.argwhere(np.sum(m1, axis=1) != 0)
        y_min = max(m1_y[0][0] - 2, 0)
        y_max = min(m1_y[-1][0] + 3, shp[0])
        m1_y = None

        # X extent
        m1_x = np.argwhere(np.sum(m1, axis=0) != 0)
        x_min = max(m1_x[0][0] - 2, 0)
        x_max = min(m1_x[-1][0] + 3, shp[1])
        m1_x = None
        m1 = None

        # Coordinate arrays
        # HRRR GRIB has data in south-up format
        trans = prof["transform"].to_gdal()
        xArr = trans[0] + (np.arange(x_min, x_max) + 0.5) * trans[1]
        yArr = trans[3] + (prof["height"] * trans[5]) - (np.arange(y_min, y_max) + 0.5) * trans[5]
        lats = lats[y_min:y_max, x_min:x_max]
        lons = lons[y_min:y_max, x_min:x_max]

        # Data variables
        t = ds['t'][:, y_min:y_max, x_min:x_max].to_numpy()
        z = ds['gh'][:, y_min:y_max, x_min:x_max].to_numpy()
        q = ds['q'][:, y_min:y_max, x_min:x_max].to_numpy()
        ds.close()

        # This section is purely for flipping arrays as needed
        # to match ECMWF reader is doing
        # All flips are views - no extra memory use
        # Lon -> From west to east
        # Lat -> From south to north (ECMWF reads north to south and flips it
        # load_weather) - So we do south to north here
        # Pres -> High to Los - (ECWMF does now to high and flips it back) - so
        # we do high to low
        # Data is currently in [levels, y, x] order
        flip_axes = []
        if levels[-1] > levels[0]:
            flip_axes.append(0)
            levels = np.flip(levels)

        if lats[0, 0] > lats[-1, 0]:
            flip_axes.append(1)
            lats = np.flip(lats, 0)
            yArr = np.flip(yArr)

        if lons[0, 0] > lons[0, -1]:
            flip_axes.append(2)
            lons = np.flip(lons, 1)
            xArr = np.flip(xArr)

        flip_axes = tuple(flip_axes)
        if flip_axes:
            t = np.flip(t, flip_axes)
            z = np.flip(z, flip_axes)
            q = np.flip(q, flip_axes)

        # Create output dataset
        ds_new = xarray.Dataset(
            data_vars=dict(
                t= (["level", "y", "x"], t,
                    {"grid_mapping": "proj"}),
                z= (["level", "y", "x"], z,
                    {"grid_mapping": "proj"}),
                q= (["level", "y", "x"], q,
                    {"grid_mapping": "proj"}),
                lats=(["y", "x"], lats),
                lons=(["y", "x"], lons),
            ),
            coords=dict(
                levels=(["level"], levels,
                       {"units": "millibars",
                        "long_name":  "pressure_level",
                        "axis": "Z"}),
                x=(["x"], xArr,
                   {"standard_name": "projection_x_coordinate",
                    "units": "m",
                    "axis": "X"}),
                y=(["y"], yArr,
                   {"standard_name": "projection_y_coordinate",
                    "units": "m",
                    "axis": "Y"}),
            ),
            attrs={
                'Conventions': 'CF-1.7',
                'Weather_model':'HRRR',
           }
        )

        # Write projection of output
        ds_new["proj"] = int()
        for k, v in self._proj.to_cf().items():
            ds_new.proj.attrs[k] = v

        ds_new.to_netcdf(out, engine='netcdf4')


    def _download_hrrr_file(self, DATE, out, model='hrrr', product='prs', fxx=0, verbose=False):
        '''
        Download a HRRR model
        '''
        ## TODO: Check how Herbie does temporal interpolation
        # corrected_DT = round_date(DATE, datetime.timedelta(hours=self._time_res))
        # if not corrected_DT == DATE:
        #     logger.warning('Rounded given datetime from  %s to %s', DATE, corrected_DT)
        corrected_DT = DATE

        H = Herbie(
            corrected_DT.strftime('%Y-%m-%d %H:%M'),
            model=model,
            product=product,
            overwrite=False,
            verbose=True,
            save_dir=Path(os.path.dirname(out)),
        )
        pf = H.download(":(SPFH|PRES|TMP|HGT):", verbose=verbose)

        self._makeDataCubes(pf, out)

        return out
