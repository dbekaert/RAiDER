import datetime
import glob
import os
import operator
import pytest

import numpy as np

from functools import reduce
from numpy import nan
from osgeo import gdal
from test import DATA_DIR, TEST_DIR, pushd

from RAiDER.constants import Zenith, _ZMIN, _ZREF
from RAiDER.processWM import prepareWeatherModel
from RAiDER.models.weatherModel import (
    WeatherModel,
    find_svp,
    make_raw_weather_data_filename,
    make_weather_model_filename,
)
from RAiDER.models.erai import ERAI
from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.hres import HRES
from RAiDER.models.hrrr import HRRR
from RAiDER.models.gmao import GMAO
from RAiDER.models.merra2 import MERRA2
from RAiDER.models.ncmr import NCMR


WEATHER_FILE = os.path.join(
    DATA_DIR,
    "weather_files",
    "ERA-5_2018_07_01_T00_00_00.nc"
)


@pytest.fixture
def erai():
    wm = ERAI()
    return wm


@pytest.fixture
def era5():
    wm = ERA5()
    return wm


@pytest.fixture
def era5t():
    wm = ERA5T()
    return wm


@pytest.fixture
def hres():
    wm = HRES()
    return wm


@pytest.fixture
def gmao():
    wm = GMAO()
    return wm


@pytest.fixture
def merra2():
    wm = MERRA2()
    return wm


@pytest.fixture
def hrrr():
    wm = HRRR()
    return wm


@pytest.fixture
def ncmr():
    wm = NCMR()
    return wm


def product(iterable):
    return reduce(operator.mul, iterable, 1)


class MockWeatherModel(WeatherModel):
    """Implement abstract methods for testing."""

    def __init__(self):
        super().__init__()

        self._Name = "MOCK"
        self._valid_range = (datetime.datetime(1970, 1, 1), "Present")
        self._lag_time = datetime.timedelta(days=15)

    def _fetch(self, lats, lons, time, out):
        pass

    def load_weather(self, *args, **kwargs):
        pass


@pytest.fixture
def model():
    return MockWeatherModel()


def test_weatherModel_basic1(model):
    wm = model
    assert wm._zmin == _ZMIN
    assert wm._zmax == _ZREF
    assert wm.Model() == 'MOCK'

    # check some defaults
    assert wm._humidityType == 'q'

    wm.setTime(datetime.datetime(2020, 1, 1, 6, 0, 0))
    assert wm._time == datetime.datetime(2020, 1, 1, 6, 0, 0)

    wm.setTime('2020-01-01T00:00:00')
    assert wm._time == datetime.datetime(2020, 1, 1, 0, 0, 0)

    wm.setTime('19720229', fmt='%Y%m%d')  # test a leap year
    assert wm._time == datetime.datetime(1972, 2, 29, 0, 0, 0)

    with pytest.raises(RuntimeError):
        wm.checkTime(datetime.datetime(1950, 1, 1))

    wm.checkTime(datetime.datetime(2000, 1, 1))

    with pytest.raises(RuntimeError):
        wm.checkTime(datetime.datetime.now())


def test_uniform_in_z_small(model):
    # Uneven z spacing, but averages to [1, 2]
    model._zs = np.array([
        [[1., 2.],
         [0.9, 1.1]],

        [[1., 2.6],
         [1.1, 2.3]]
    ])
    model._xs = np.array([1, 2, 2])
    model._ys = np.array([2, 3, 2])
    model._p = np.arange(8).reshape(2, 2, 2)
    model._t = model._p * 2
    model._e = model._p * 3
    model._lats = model._zs  # for now just passing in dummy arrays for lats, lons
    model._lons = model._zs

    model._uniform_in_z()

    # Note that when the lower bound is exactly equal we get a value, but
    # when the upper bound is exactly equal we get the fill
    interpolated = np.array([
        [[0, nan],
         [2.5, nan]],

        [[4., 4.625],
         [nan, 6.75]]
    ])

    assert np.allclose(model._p, interpolated, equal_nan=True, rtol=0)
    assert np.allclose(model._t, interpolated * 2, equal_nan=True, rtol=0)
    assert np.allclose(model._e, interpolated * 3, equal_nan=True, rtol=0)

    assert np.allclose(model._zs, np.array([1, 2]), rtol=0)
    assert np.allclose(model._xs, np.array([1, 2]), rtol=0)
    assert np.allclose(model._ys, np.array([2, 3]), rtol=0)


def test_uniform_in_z_large(model):
    shape = (400, 500, 40)
    x, y, z = shape
    size = product(shape)

    # Uneven spacing that averages to approximately [1, 2, ..., 39, 40]
    zlevels = np.arange(1, shape[-1] + 1)
    model._zs = np.random.normal(1, 0.1, size).reshape(shape) * zlevels
    model._xs = np.empty(0)  # Doesn't matter
    model._ys = np.empty(0)  # Doesn't matter
    model._p = np.tile(np.arange(y).reshape(-1, 1) * np.ones(z), (x, 1, 1))
    model._t = model._p * 2
    model._e = model._p * 3
    model._lats = model._zs  # for now just passing in dummy arrays for lats, lons
    model._lons = model._zs

    assert model._p.shape == shape
    model._uniform_in_z()

    interpolated = np.tile(np.arange(y), (x, 1))

    assert np.allclose(np.nanmean(model._p, axis=-1),
                       interpolated, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._t, axis=-1),
                       interpolated * 2, equal_nan=True, rtol=0)
    assert np.allclose(np.nanmean(model._e, axis=-1),
                       interpolated * 3, equal_nan=True, rtol=0)

    assert np.allclose(model._zs, zlevels, atol=0.05, rtol=0)


def test_mwmf():
    name = 'ERA-5'
    time = datetime.datetime(2020, 1, 1)
    ll_bounds = (-90, 90, -180, 180)
    assert make_weather_model_filename(name, time, ll_bounds) == \
        'ERA-5_2020_01_01_T00_00_00_90S_90N_180W_180E.nc'


def test_mrwmf():
    outLoc = './'
    name = 'ERA-5'
    time = datetime.datetime(2020, 1, 1)
    assert make_raw_weather_data_filename(outLoc, name, time) == \
        './ERA-5_2020_01_01_T00_00_00.nc'


def test_checkLL_era5(era5):
    lats_good = np.array([-89, -45, 0, 45, 89])
    lons_good = np.array([-179, -90, 0, 90, 179])
    lats = np.array([-90, -45, 0, 45, 90])
    lons = np.array([-180, -90, 0, 90, 180])
    lats2, lons2 = era5.checkLL(lats, lons)
    assert np.allclose(lats2, lats)
    assert np.allclose(lons2, lons)


def test_checkLL_era5_2(era5):
    lats_good = np.array([-89, -45, 0, 45, 89])
    lons_good = np.array([-179, -90, 0, 90, 179])
    lats = np.array([-95, -45, 0, 45, 90])
    lons = np.array([-180, -90, 0, 90, 200])
    lats2, lons2 = era5.checkLL(lats, lons)
    assert np.allclose(lats2, lats_good)
    assert np.allclose(lons2, lons_good)


def test_erai(erai):
    wm = erai
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-I'
    assert wm._valid_range[0] == datetime.datetime(1979, 1, 1)
    assert wm._valid_range[1] == datetime.datetime(2019, 8, 31)
    assert wm._proj.to_epsg() == 4326


def test_era5(era5):
    wm = era5
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_era5t(era5t):
    wm = era5t
    assert wm._humidityType == 'q'
    assert wm._Name == 'ERA-5T'
    assert wm._valid_range[0] == datetime.datetime(1950, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hres(hres):
    wm = hres
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRES'
    assert wm._valid_range[0] == datetime.datetime(1983, 4, 20)
    assert wm._proj.to_epsg() == 4326
    assert wm._levels == 137

    wm.update_a_b()
    assert wm._levels == 91


def test_gmao(gmao):
    wm = gmao
    assert wm._humidityType == 'q'
    assert wm._Name == 'GMAO'
    assert wm._valid_range[0] == datetime.datetime(2014, 2, 20)
    assert wm._proj.to_epsg() == 4326


def test_merra2(merra2):
    wm = merra2
    assert wm._humidityType == 'q'
    assert wm._Name == 'MERRA2'
    assert wm._valid_range[0] == datetime.datetime(1980, 1, 1)
    assert wm._proj.to_epsg() == 4326


def test_hrrr(hrrr):
    wm = hrrr
    assert wm._humidityType == 'q'
    assert wm._Name == 'HRRR'
    assert wm._valid_range[0] == datetime.datetime(2016, 7, 15)
    assert wm._proj.to_epsg() is None


def test_ncmr(ncmr):
    wm = ncmr
    assert wm._humidityType == 'q'
    assert wm._Name == 'NCMR'
    assert wm._valid_range[0] == datetime.datetime(2015, 12, 1)


def test_find_svp():
    t = np.arange(0, 100, 10) + 273.15
    svp_test = find_svp(t)
    svp_true = np.array([
        611.21, 1227.5981, 2337.2825, 4243.5093,
        7384.1753, 12369.2295, 20021.443, 31419.297,
        47940.574, 71305.16
    ])
    assert np.allclose(svp_test, svp_true)
