import os
import pytest

import numpy as np

import pyproj
from test import DATA_DIR, pushd

from RAiDER.utilFcns import writePnts2HDF5
from RAiDER.delay import (
    checkQueryPntsFile,
    getZTD,
    getTransformedPoints,
)


@pytest.fixture
def pnts():
    lats = (10, 12)
    lons = (-72, -74)
    heights = (0, 0)
    return lats, lons, heights

@pytest.fixture
def pnts_file():
    lats = np.array([10, 12])
    lons = np.array([-72, -74])
    heights = np.array([0, 0])
    pnts = np.stack([lats, lons, heights], axis=-1)
    los = np.array([[0, 0], [0, 0], [1, 1]])
    filename = 'query_points_test_temp.h5'
    writePnts2HDF5(
            lats, 
            lons, 
            hgts, 
            los, 
            filename, 
            noDataValue = -9999
        )
    return filename, pnts


def test_cqpf1():
    assert checkQueryPntsFile('does_not_exist.h5', None)

def test_cqpf2(pnts_file):
    filename, pnts = pnts_file
    assert ~checkQueryPntsFile(filename, (1,2))

def test_cqpf3(pnts_file):
    filename, pnts = pnts_file
    assert checkQueryPntsFile(filename, (2,1))


def test_getZTD(wm_file):
    pass

def test_getTransformedPoints(pnts):
    lats, lons, heights = pnts
    projection = pyproj.Proj(proj='geocent')
    tpnts = getTransformedPoints(lats, lons, heights, projection)
    tru_points = np.array([])
    assert np.allclose(tpnts, tru_points)
