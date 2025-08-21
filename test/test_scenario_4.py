import datetime as dt
from pathlib import Path
from typing import Type

import numpy as np
import pytest
import xarray
from pyproj import CRS

from RAiDER.delay import _get_delays_on_cube, tropo_delay
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith
from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.gmao import GMAO
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.ncmr import NCMR
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.processWM import prepareWeatherModel
from test import TEST_DIR, pushd


SCENARIO_DIR = TEST_DIR / 'scenario_4'

DATETIME = dt.datetime(2020, 1, 1)


@pytest.mark.long
@pytest.mark.parametrize('Model', (GMAO, NCMR, MERRA2, ERA5T, ERA5))
def test_aoi_without_xpts(tmp_path: Path, Model: Type[WeatherModel]) -> None:
    with pushd(tmp_path):
        los = Zenith()
        latfile = SCENARIO_DIR / 'lat.rdr'
        lonfile = SCENARIO_DIR / 'lon.rdr'
        hgtfile = SCENARIO_DIR / 'hgt.rdr'
        aoi = RasterRDR(latfile, lonfile, hgtfile)

        wm = Model()
        wm.set_latlon_bounds(aoi.bounds())
        wm.setTime(DATETIME)
        f = prepareWeatherModel(wm, DATETIME, aoi.bounds())
        zen_wet, zen_hydro = tropo_delay(DATETIME, f, aoi, los)

        assert len(zen_wet.shape) == 2
        assert np.sum(np.isnan(zen_wet)) < np.prod(zen_wet.shape)
        assert np.nanmean(zen_wet) > 0
        assert np.nanmean(zen_hydro) > 0


@pytest.mark.long
@pytest.mark.parametrize('Model', (GMAO, NCMR, MERRA2, ERA5T, ERA5))
def test_get_delays_on_cube(tmp_path: Path, Model: Type[WeatherModel]) -> None:
    with pushd(tmp_path):
        los = Zenith()
        latfile = SCENARIO_DIR / 'lat.rdr'
        lonfile = SCENARIO_DIR / 'lon.rdr'
        hgtfile = SCENARIO_DIR / 'hgt.rdr'
        aoi = RasterRDR(latfile, lonfile, hgtfile)

        wm = Model()
        wm.set_latlon_bounds(aoi.bounds())
        wm.setTime(DATETIME)
        f = prepareWeatherModel(wm, DATETIME, aoi.bounds())

        with xarray.load_dataset(f) as ds:
            wm_levels = ds['z'].values
            wm_proj = CRS.from_wkt(ds['proj'].attrs['crs_wkt'])

        zref = 10000

        with pytest.raises(AttributeError):
            aoi.xpts

        ds = _get_delays_on_cube(DATETIME, f, wm_proj, aoi, wm_levels, los, wm_proj, zref)

        assert len(ds.x) > 0
        assert ds['hydro'].mean() > 0
