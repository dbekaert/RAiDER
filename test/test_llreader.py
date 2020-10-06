import os
import pytest

import numpy as np

from test import GEOM_DIR, TEST_DIR

from RAiDER.llreader import (
    readLLFromLLFiles,readLLFromBBox,readLLFromStationFile,forceNDArray
)

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_2")

def test_latlon_reader():
    lats, lons, llproj = readLLFromLLFiles(os.path.join(GEOM_DIR, 'lat.rdr'), os.path.join(GEOM_DIR, 'lon.rdr'))
    assert lats.shape == (45, 226)
    assert lons.shape == (45, 226)
    

def test_bbox_reade1():
    lat, lon, llproj = readLLFromBBox(['10', '12', '-72', '-70'])
    print(lat)
    print(lon)
    assert np.allclose(lat, np.array([10, 12]))
    assert np.allclose(lon, np.array([-72, -70]))

def test_stationfile_reader():
    lats, lons, llproj = readLLFromStationFile(os.path.join(SCENARIO_DIR, 'stations.csv'))
    assert len(lats)==8

def test_forceNDArray():
    assert np.all(np.array([1, 2, 3]) == forceNDArray([1, 2, 3]))
    assert np.all(np.array([1, 2, 3]) == forceNDArray((1, 2, 3)))
    assert forceNDArray(None) is None

