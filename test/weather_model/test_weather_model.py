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

from RAiDER.constants import Zenith
from RAiDER.processWM import prepareWeatherModel
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.models.era5 import ERA5

WEATHER_FILE = os.path.join(
    DATA_DIR,
    "weather_files",
    "ERA-5_2018_07_01_T00_00_00.nc"
)


@pytest.fixture
def era5():
    era5_wm = ERA5()
    return era5_wm

WEATHER_FILE = 0# TODO

@pytest.fixture
def era5():
    lats = np.arange(20, 20.5, 0.1)
    lons = np.arrange(-73, -72.5, 0.1)
    [lats, lons] = np.meshgrid(lat, lon)
    time = datetime.datetime(2020,1,1,0,0,0)
    era5_wm = ERA5()
    era5_wm.load(WEATHER_FILE, lats, lons, los = Zenith, zref = 20000)
    return era5_wm
    
def product(iterable):
    return reduce(operator.mul, iterable, 1)

class MockWeatherModel(WeatherModel):
    """Implement abstract methods for testing."""

    def __init__(self):
        super().__init__()

        self._Name = "MOCK"

    def _fetch(self, lats, lons, time, out):
        pass

    def load_weather(self, *args, **kwargs):
        pass


@pytest.fixture
def model():
    return MockWeatherModel()


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

    model._uniform_in_z()

    interpolated = np.array([
        # Note that when the lower bound is exactly equal we get a value, but
        # when the upper bound is exactly equal we get the fill
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
