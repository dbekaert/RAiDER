import datetime 
import os
import pytest
import xarray

import numpy as np
from pyproj import CRS

from RAiDER.delay import tropo_delay, _get_delays_on_cube
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith
from RAiDER.processWM import prepareWeatherModel
from RAiDER.models.era5 import ERA5

from test import TEST_DIR
SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_4")


def test_aoi_without_xpts():
    los = Zenith()
    latfile = os.path.join(SCENARIO_DIR, 'lat.rdr')
    lonfile = os.path.join(SCENARIO_DIR, 'lon.rdr')
    hgtfile = os.path.join(SCENARIO_DIR, 'hgt.rdr')
    aoi = RasterRDR(latfile, lonfile, hgtfile)
    dt = datetime.datetime(2020,1,1)

    wm = ERA5()
    wm.set_latlon_bounds(aoi.bounds())
    wm.setTime(dt)
    f = prepareWeatherModel(wm, dt, aoi.bounds())

    zen_wet, zen_hydro = tropo_delay(dt, f, aoi, los)

    assert len(zen_wet.shape) == 2
    assert np.sum(np.isnan(zen_wet)) < np.prod(zen_wet.shape)
    assert np.nanmean(zen_wet) > 0
    assert np.nanmean(zen_hydro) > 0


def test_get_delays_on_cube():

    los = Zenith()
    latfile = os.path.join(SCENARIO_DIR, 'lat.rdr')
    lonfile = os.path.join(SCENARIO_DIR, 'lon.rdr')
    hgtfile = os.path.join(SCENARIO_DIR, 'hgt.rdr')
    aoi = RasterRDR(latfile, lonfile, hgtfile)
    dt = datetime.datetime(2020,1,1)

    wm = MERRA2()
    wm.set_latlon_bounds(aoi.bounds())
    wm.setTime(dt)
    f = prepareWeatherModel(wm, dt, aoi.bounds())

    with xarray.load_dataset(f) as ds:
        wm_levels = ds.z.values
        wm_proj = CRS.from_wkt(ds['proj'].attrs['crs_wkt'])
    
    zref = 10000

    with pytest.raises(AttributeError):
        aoi.xpts
    
    ds = _get_delays_on_cube(dt, f, wm_proj, aoi, wm_levels, los, wm_proj, zref)

    assert len(ds.x) > 0
    assert ds.hydro.mean() > 0
    
