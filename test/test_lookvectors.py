import numpy as np
import pytest

from RAiDER.rayGenerator import getLookVectors
from RAiDER.constants import Zenith
from RAiDER.utilFcns import gdal_open, enu2ecef, sind, cosd
from RAiDER.rayGenerator import Points, Zenith, Slant

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
def read_state_vector():
    return read_ESA_Orbit_file('test_geom/S1A_OPER_AUX_POEORB_OPOD_20200122T120701_V20200101T225942_20200103T005942.EOF')

def test_Points_1():
    pts = Points()
    pts.setLLH(np.random.randn((10,3)))
    pts._checkLLH()

def test_Points_2():
    pts = Points()
    pts.setLLH(np.random.randn((10,10,3)))
    pts._checkLLH()

def test_Points_3():
    pts = Points()
    pts.setLLH(np.random.randn((10,10,10,3)))
    pts._checkLLH()

def test_Points_4(make_llh):
    pts = Points()
    pts.setLLH(make_llh)
    pts._checkLLH()


