import os
import pytest

import numpy as np

import pyproj
from test import DATA_DIR, pushd

from RAiDER.utilFcns import writePnts2HDF5
from RAiDER.delay import (
    checkQueryPntsFile,
    transformPoints,
)


@pytest.fixture
def pnts():
    lats = np.array([10, 12])
    lons = np.array([-72, -74])
    heights = np.array([0, 0])
    return lats, lons, heights


@pytest.fixture
def pnts_file():
    lats = np.array([10, 12])
    lons = np.array([-72, -74])
    hgts = np.array([0, 0])
    pnts = np.stack([lats, lons, hgts], axis=-1)
    los = np.array([[0, 0], [0, 0], [1, 1]]).T
    lengths = np.array([100, 100])
    filename = 'query_points_test_temp.h5'
    writePnts2HDF5(
        lats,
        lons,
        hgts,
        los,
        lengths,
        filename,
        noDataValue=-9999
    )
    return filename, pnts


def test_cqpf1(pnts_file):
    assert checkQueryPntsFile('does_not_exist.h5', None)


def test_cqpf2(pnts_file):
    filename, pnts = pnts_file
    assert ~checkQueryPntsFile(filename, (1, 2))


def test_cqpf3(pnts_file):
    filename, pnts = pnts_file
    assert checkQueryPntsFile(filename, (2, 1))


def test_transformPoints(pnts):
    lats, lons, heights = pnts
    old = pyproj.crs.CRS(4326)
    new = pyproj.crs.CRS(4978)
    tpnts = transformPoints(lats, lons, heights, old, new)
    tru_points = np.array([[1941205.46084971, -5974416.08913184, 1100248.54773536],
                           [1719884.01344839, -5997948.350231, 1317402.5312296]]).T
    assert np.allclose(tpnts, tru_points)
