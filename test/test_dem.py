import os
import pytest

import numpy as np

from test import DATA_DIR, pushd

from RAiDER.dem import (
    getBufferedExtent, isOutside, isInside, getDEM, openDEM, forceNDArray
)


@pytest.fixture
def llsimple():
    lats = (10, 12)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def latwrong():
    lats = (12, 10)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def lonwrong():
    lats = (10, 12)
    lons = (-72, -74)
    return lats, lons


@pytest.fixture
def llarray():
    lats = np.arange(10, 12.1, 0.1)
    lons = np.arange(-74, -71.9, 0.2)
    return lats, lons


def test_ll1(llsimple):
    lats, lons = llsimple
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll2(latwrong):
    lats, lons = latwrong
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll3(lonwrong):
    lats, lons = lonwrong
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_ll4(llarray):
    lats, lons = llarray
    assert np.allclose(getBufferedExtent(lats, lons), np.array([10, 12, -74, -72]))


def test_isOutside1(llsimple):
    assert isOutside(getBufferedExtent(*llsimple), getBufferedExtent(*llsimple) + 1)


def test_isOutside2(llsimple):
    assert not isOutside(getBufferedExtent(*llsimple), getBufferedExtent(*llsimple))


def test_isInside(llsimple):
    assert isInside(getBufferedExtent(*llsimple), getBufferedExtent(*llsimple))


def test_isInside(llsimple):
    assert not isInside(getBufferedExtent(*llsimple), getBufferedExtent(*llsimple) + 1)


def test_getDEM(tmp_path):
    with pushd(tmp_path):
        getDEM([18.5, 18.9, -73.2, -72.8], tmp_path)

def test_openDEM():
    dem = openDEM(os.path.join(DATA_DIR, 'geom'), dem_raster = 'warpedDEM.dem')
    assert dem.size >0
    assert dem.ndim == 2


def test_isNDArray():
    assert np.allclose(forceNDArray(np.ones((10,))), np.ones((10,)))
    assert np.allclose(forceNDArray(np.empty((10,))), np.empty((10,)))
    assert np.allclose(forceNDArray([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]))
    assert forceNDArray(None) is None
