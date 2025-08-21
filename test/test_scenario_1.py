import datetime as dt
from pathlib import Path
from typing import Type

import numpy as np
import pytest

from RAiDER.delay import tropo_delay
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith
from RAiDER.models import ERA5, ERA5T, GMAO, MERRA2
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import rio_open
from test import TEST_DIR


SCENARIO_DIR = TEST_DIR / 'scenario_1'
WM_LOC = SCENARIO_DIR / 'weather_files'

_RTOL = 1e-2
DATETIME = dt.datetime(2018, 7, 1, 0, 0)
DATETIME_GMAO_OLD = dt.datetime(2017, 1, 1, 0, 0, 0)


@pytest.mark.long
@pytest.mark.parametrize(
    'Model,time',
    (
        pytest.param(ERA5, DATETIME, id='ERA5'),
        pytest.param(ERA5T, DATETIME, id='ERA5T'),
        pytest.param(GMAO, DATETIME, id='GMAO new'),
        pytest.param(GMAO, DATETIME_GMAO_OLD, id='GMAO old'),
        pytest.param(MERRA2, DATETIME, id='MERRA2'),
    ),
)
def test_tropo_delay(tmp_path: Path, Model: Type[WeatherModel], time: dt.datetime) -> None:
    """
    Scenario:
    1: Small area, Zenith delay.
    """
    WM_LOC.mkdir(exist_ok=True)

    los = Zenith()
    lat_path = str(SCENARIO_DIR / 'geom/lat.dat')
    lon_path = str(SCENARIO_DIR / 'geom/lon.dat')
    hgt_file = str(TEST_DIR / 'test_geom/warpedDEM.rdr')
    aoi = RasterRDR(lat_path, lon_path, hgt_file=hgt_file, output_directory=str(tmp_path))

    wm = Model()
    wm.set_latlon_bounds(aoi.bounds())
    wm.setTime(time)
    wm.set_wmLoc(str(WM_LOC))
    wm_file_path = prepareWeatherModel(wm, time, aoi.bounds())
    wet, hydro = tropo_delay(time, wm_file_path, aoi, los, zref=20_000)

    # load the true delay
    true_wet, _ = rio_open(SCENARIO_DIR / f'{wm._Name}/wet.envi', userNDV=0.0)
    true_hydro, _ = rio_open(SCENARIO_DIR / f'{wm._Name}/hydro.envi', userNDV=0.0)

    assert np.nanmax(np.abs((wet - true_wet) / true_wet)) < _RTOL
    assert np.nanmax(np.abs((hydro - true_hydro) / true_hydro)) < _RTOL
