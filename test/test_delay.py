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

def test_transformPoints(pnts):
    lats, lons, heights = pnts
    old = pyproj.crs.CRS(4326)
    new = pyproj.crs.CRS(4978)
    tpnts = transformPoints(lats, lons, heights, old, new).T
    tru_points = np.array([[ 6144598.8363915 ,   544920.48311418,  6378137.        ],
       [ 1306074.801505  , -1900363.56355715,        0.        ]])
    assert np.allclose(tpnts, tru_points)

