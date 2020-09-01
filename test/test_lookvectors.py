from test import TEST_DIR

import numpy as np
import pytest

from RAiDER.losreader import read_ESA_Orbit_file, read_los_file
from RAiDER.rays import (
    OrbitLVGenerator, ZenithLVGenerator
)
from RAiDER.utilFcns import gdal_open
from RAiDER.cli.validators import incidence_heading_to_los

LOS_FILE = TEST_DIR / "test_geom" / "los.rdr"


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
    return read_ESA_Orbit_file(TEST_DIR / 'test_geom' / 'S1A_OPER_AUX_POEORB_OPOD_20200122T120701_V20200101T225942_20200103T005942.EOF')


@pytest.fixture
def zenithgen():
    return ZenithLVGenerator()


@pytest.fixture
def orbitgen(state_vector):
    return OrbitLVGenerator(state_vector)


def test_zenith_generator_simple(zenithgen):
    _2 = 1 / np.sqrt(2)
    _3 = 1 / np.sqrt(3)

    llh = np.array([
        [0., 0., 0.],  # zenith is x-axis
        [0., 0., 0.],
        [0., 90., 0.],  # zenith is y-axis
        [90., 0., 0.],  # zenith is z-axis
        [45., 90., 0.],
        [45., 0., 0.],
        [0., 45., 0.],
        [90. - np.degrees(np.arccos(_3)), 45., 0.],  # zenith is the line x=y=z
    ])
    ans = zenithgen.generate(llh)

    assert np.allclose(ans, np.array([
        [1., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.],
        [0., _2, _2],
        [_2, 0., _2],
        [_2, _2, 0.],
        [_3, _3, _3],
    ]))


def test_orbit_generator_simple(orbitgen):
    llh = np.array([
        [40, -80., 0.],
        [40, -85., 0.],
        [40, -75., 0.],
        [45, -85., 0.],
        [45, -80., 0.],
        [45, -75., 0.],
        [35, -85., 0.],
        [35, -80., 0.],
        [35, -75., 0.],
    ])
    ans = orbitgen.generate(llh)

    # These values are copy/pasted generated values from the above call. We
    # visually plotted them and they looked right, so we assume they are and
    # use this as a regression test so we know if the behaviour of the function
    # is changed.
    assert np.allclose(ans, np.array([
        [-0.05146681, -0.80103483, 0.59640119],
        [+0.45607286, -0.58576985, 0.66997853],
        [-0.46484434, -0.79222125, 0.39535456],
        [+0.26247899, -0.63102352, 0.73000966],
        [-0.22526362, -0.76458368, 0.60387756],
        [-0.54500409, -0.73164134, 0.40947709],
        [+0.55564835, -0.56916693, 0.60605603],
        [+0.12338176, -0.81340164, 0.56846699],
        [-0.26466444, -0.86499673, 0.42630199],
    ]))


@pytest.fixture(params=('zenithgen', 'orbitgen'))
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

    magnitude = np.linalg.norm(ans, axis=-1)
    assert np.allclose(magnitude, 1.)


def test_incidence_heading_to_los_returns_unit_vectors():
    los = incidence_heading_to_los(*read_los_file(LOS_FILE))

    magnitude = np.linalg.norm(los, axis=-1)
    assert np.allclose(magnitude[~np.isnan(magnitude)], 1.)
