import numpy as np
import pytest

from RAiDER.rayGenerator import getLookVectors
from RAiDER.constants import Zenith
from RAiDER.utilFcns import gdal_open, enu2ecef, sind, cosd

@pytest.fixture
def read_los():
    return gdal_open('test_geom/los.rdr')

@pytest.fixture
def read_hgt():
    return gdal_open('test_geom/warpedDEM.dem')
    
@pytest.fixture
def read_lat():
    return gdal_open('test_geom/lat.rdr')
    
@pytest.fixture
def read_lon():
    return gdal_open('test_geom/lon.rdr')

def make_llh(read_lat, read_lon, read_hgt):
    return np.stack([read_lat, read_lon, read_hgt], axis=-1)

@pytest.fixture
def make_los(read_los, read_lon, read_lat):
    los = read_los
    lon = read_lon
    lat = read_lat

    inc = los[...,0]
    hd = los[...,1]
    east = sind(inc) * cosd(hd + 90)
    north = sind(inc) * sind(hd + 90)
    up = cosd(inc)
    xyz = enu2ecef(east, north, up, lats, lons, heights)
    return xyz

@pytest.fixture
def read_state_vector():
    return *read_ESA_Orbit_file('test_geom/S1A_OPER_AUX_POEORB_OPOD_20200122T120701_V20200101T225942_20200103T005942.EOF')

def test_getLookVectors_Zenith():
    lvs = np.zeros((3,3))
    lvs[:,-1] = 1
    assert getLookVectors(Zenith, np.arange(10), np.arange(10), np.arange(10)) == lvs

def test_getLookVectors_losFile(make_los, make_llh):
    los = make_los
    llh = make_llh

    assert getLookVectors('test_geom/los.rdr', llh) == los

def test_getLookVectors_EOFStateFile(read_state_vector, make_llh):
    t, x, y, z, vx, vy, vz = read_state_vector
    llh = make_llh

    assert getLookVectors('test_geom/los.rdr', llh) == los
