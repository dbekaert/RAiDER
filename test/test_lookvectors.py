from test import TEST_DIR

import numpy as np
import pytest

from RAiDER.losreader import read_ESA_Orbit_file
from RAiDER.rays import Points
from RAiDER.utilFcns import gdal_open



@pytest.fixture
def los():
    return gdal_open(TEST_DIR / "test_geom" / "los.rdr")



@pytest.fixture
def lat():
    return gdal_open(TEST_DIR / "test_geom" / "lat.rdr")


@pytest.fixture
def lon():
    return gdal_open(TEST_DIR / "test_geom" / "lon.rdr")


@pytest.fixture
def hgt():
    return gdal_open(TEST_DIR / "test_geom" / "warpedDEM.dem")


@pytest.fixture
def llh(lat, lon, hgt):
    return np.stack([lat, lon, hgt], axis=-1)


@pytest.fixture
def state_vector():
    return read_ESA_Orbit_file('test_geom/S1A_OPER_AUX_POEORB_OPOD_20200122T120701_V20200101T225942_20200103T005942.EOF')


def test_Points_1():
    pts = Points(np.random.randn(10, 3))


def test_Points_2():
    pts = Points(np.random.randn(10, 10, 3))


def test_Points_3():
    pts = Points(np.random.randn(10, 10, 10, 3))


def test_Points_4(llh):
    pts = Points(llh)


def test_zenith_generator():
    pass
