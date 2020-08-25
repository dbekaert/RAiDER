from test import TEST_DIR

import numpy as np
import pytest
from numpy import testing as npt

from RAiDER.losreader import read_ESA_Orbit_file
from RAiDER.rays import Points, ZenithLVGenerator
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


@pytest.fixture
def zenithgen():
    return ZenithLVGenerator()


def test_Points_1():
    pts = Points(np.random.randn(10, 3))


def test_Points_2():
    pts = Points(np.random.randn(10, 10, 3))


def test_Points_3():
    pts = Points(np.random.randn(10, 10, 10, 3))


def test_Points_4(llh):
    pts = Points(llh)


def test_zenith_generator_simple(zenithgen):
    llh = np.array([
        [0., 0., 0.],  # zenith is x-axis
        [0., 0., 0.],
        [0., 90., 0.],  # zenith is y-axis
        [90., 0., 0.],  # zenith is z-axis
        [45., 90., 0.],
        [45., 0., 0.],
        [0., 45., 0.],
    ])
    ans = zenithgen.generate(llh)

    _2 = 1 / np.sqrt(2)
    assert np.allclose(ans, np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., _2, _2],
        [_2, 0., _2],
        [_2, _2, 0.],
    ]))


@pytest.fixture(params=('zenithgen', ))
def lvgen_param(request):
    """A bit of a hack to parameterize a test by fixture"""
    return request.getfixturevalue(request.param)


def test_generator_returns_unit_vectors(lvgen_param):
    gen = lvgen_param

    lats = np.random.rand(100, 100) * 180 - 90
    lons = np.random.rand(100, 100) * 360 - 180
    heights = np.random.rand(100, 100) * 20000
    llh = np.stack((lats, lons, heights), axis=-1)

    ans = gen.generate(llh)

    magnitude = np.sqrt(ans[..., 0] ** 2 + ans[..., 1] ** 2 + ans[..., 2] ** 2)
    assert np.allclose(magnitude, 1.)
