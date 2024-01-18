import datetime 
import os
import pytest

import numpy as np

from RAiDER.delay import tropo_delay
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith
from RAiDER.processWM import prepareWeatherModel
from RAiDER.models.era5 import ERA5

from test import TEST_DIR
SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_4")


@pytest.mark.long
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



