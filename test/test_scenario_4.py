import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from pyproj import CRS

from RAiDER.delay import _get_delays_on_cube, tropo_delay
from RAiDER.llreader import RasterRDR
from RAiDER.losreader import Zenith
from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.gmao import GMAO
from RAiDER.models.hres import HRES
from RAiDER.models.hrrr import HRRR
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.processWM import prepareWeatherModel
from test import TEST_DIR


SCENARIO_DIR = TEST_DIR / 'scenario_4'

DATETIME = dt.datetime(2020, 1, 1)
ZREF = 10_000


@pytest.mark.long
@pytest.mark.parametrize(
    'Model',
    (
        pytest.param(ERA5),
        pytest.param(ERA5T),
        pytest.param(GMAO, marks=pytest.mark.skip),  # TODO: dbekaert/RAiDER#755
        pytest.param(HRES, marks=pytest.mark.skip),  # Paid model
        pytest.param(HRRR, marks=pytest.mark.skip),  # TODO
        pytest.param(MERRA2),
    ),
)
def test_aoi_without_xpts(tmp_path: Path, Model: type[WeatherModel]) -> None:
    los = Zenith()
    latfile = SCENARIO_DIR / 'lat.rdr'
    lonfile = SCENARIO_DIR / 'lon.rdr'
    hgtfile = SCENARIO_DIR / 'hgt.rdr'
    aoi = RasterRDR(latfile, lonfile, hgt_file=hgtfile)
    aoi.set_output_directory(tmp_path)

    wm = Model()
    wm.set_latlon_bounds(aoi.bounds())
    wm.setTime(DATETIME)
    wm.set_wmLoc(str(tmp_path))
    wm_file_path = prepareWeatherModel(wm, DATETIME, aoi.bounds())
    zen_wet, zen_hydro = tropo_delay(DATETIME, wm_file_path, aoi, los)

    assert zen_wet.ndim == 2
    assert np.sum(np.isnan(zen_wet)) < zen_wet.size
    assert np.nanmean(zen_wet) > 0
    assert np.nanmean(zen_hydro) > 0


@pytest.mark.long
@pytest.mark.parametrize(
    'Model',
    (
        pytest.param(ERA5),
        pytest.param(ERA5T),
        pytest.param(GMAO, marks=pytest.mark.skip),  # TODO: dbekaert/RAiDER#755
        pytest.param(HRES, marks=pytest.mark.skip),  # Paid model
        pytest.param(HRRR, marks=pytest.mark.skip),  # TODO
        pytest.param(MERRA2),
    ),
)
def test_get_delays_on_cube(tmp_path: Path, Model: type[WeatherModel]) -> None:
    los = Zenith()
    latfile = SCENARIO_DIR / 'lat.rdr'
    lonfile = SCENARIO_DIR / 'lon.rdr'
    hgtfile = SCENARIO_DIR / 'hgt.rdr'
    aoi = RasterRDR(latfile, lonfile, hgt_file=hgtfile)
    aoi.set_output_directory(tmp_path)

    wm = Model()
    wm.set_latlon_bounds(aoi.bounds())
    wm.setTime(DATETIME)
    wm.set_wmLoc(str(tmp_path))
    wm_file_path = prepareWeatherModel(wm, DATETIME, aoi.bounds())

    with xr.open_dataset(wm_file_path) as ds:
        wm_levels = ds['z'].values
        wm_proj = CRS.from_wkt(ds['proj'].attrs['crs_wkt'])

    assert not hasattr(aoi, 'xpts')

    ds = _get_delays_on_cube(DATETIME, wm_file_path, wm_proj, aoi, wm_levels, los, wm_proj, ZREF)

    assert len(ds['x']) > 0
    assert ds['hydro'].mean() > 0
