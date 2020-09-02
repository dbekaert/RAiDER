from datetime import datetime, timezone
from test import TEST_DIR

import numpy as np
import pytest

from RAiDER.cli.validators import incidence_heading_to_los
from RAiDER.losreader import read_ESA_Orbit_file, read_los_file
from RAiDER.rays import OrbitLVGenerator, ZenithLVGenerator
from RAiDER.utilFcns import gdal_open

LOS_FILE = TEST_DIR / "test_geom" / "los.rdr"


@pytest.fixture
def LOS():
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
    return read_ESA_Orbit_file(TEST_DIR / 'los_files' / 'test_orbit.EOF')


@pytest.fixture
def zenithgen():
    return ZenithLVGenerator()


@pytest.fixture
def orbitgen(state_vector):
    return OrbitLVGenerator(state_vector)


@pytest.fixture
def acq_time():
    return datetime(2020, 1, 2, 22, 59, 42, 0, timezone.utc)


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


def test_orbit_generator_simple(orbitgen, acq_time):
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
    ans = orbitgen.generate(llh, acq_time)

    # These values are copy/pasted generated values from the above call. We
    # visually plotted them and they looked right, so we assume they are and
    # use this as a regression test so we know if the behaviour of the function
    # is changed.
    print(ans)
    assert np.allclose(ans, np.tile(np.array(
        [-0.04775827, -0.81398979,  0.57891258]
    ), (9, 1)))


@pytest.fixture(params=('zenithgen', 'orbitgen'))
def lvgen_param(request):
    """A bit of a hack to parameterize a test by fixture"""
    return request.getfixturevalue(request.param)


def test_generator_returns_unit_vectors(lvgen_param, acq_time):
    gen = lvgen_param

    lats = np.random.rand(100, 100) * 180 - 90
    lons = np.random.rand(100, 100) * 360 - 180
    heights = np.random.rand(100, 100) * 20000
    llh = np.stack((lats, lons, heights), axis=-1)

    ans = gen.generate(llh, acq_time)

    magnitude = np.linalg.norm(ans, axis=-1)
    assert np.allclose(magnitude, 1.)


def test_incidence_heading_to_los_returns_unit_vectors():
    los = incidence_heading_to_los(*read_los_file(LOS_FILE))

    magnitude = np.linalg.norm(los, axis=-1)
    assert np.allclose(magnitude[~np.isnan(magnitude)], 1.)
