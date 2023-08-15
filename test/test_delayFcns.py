import os
import pytest

import numpy as np
import xarray as xr

from pyproj import CRS, Transformer
from test import TEST_DIR

from RAiDER.delayFcns import getInterpolators
from RAiDER.delay import transformPoints

SCENARIO1_DIR = os.path.join(TEST_DIR, "scenario_1", "golden_data")


@pytest.fixture
def hrrr_proj():
    lon0 = 262.5
    lat0 = 38.5
    lat1 = 38.5
    lat2 = 38.5
    x0 = 0
    y0 = 0
    earth_radius = 6371229
    proj = CRS(f'+proj=lcc +lat_1={lat1} +lat_2={lat2} +lat_0={lat0} +lon_0={lon0} +x_0={x0} +y_0={y0} +a={earth_radius} +b={earth_radius} +units=m +no_defs')

    return proj


@pytest.fixture
def wmdata():
    return xr.load_dataset(os.path.join(SCENARIO1_DIR, 'HRRR_tropo_20200101T120000_ztd.nc'))


def test_getInterpolators(wmdata):
    ds = wmdata
    tw, th = getInterpolators(ds, kind='pointwise')
    assert True # always pass unless an error is raised

def test_getInterpolators_2(wmdata, caplog):
    ds = wmdata
    ds['hydro'][0,0,0] = np.nan
    # with pytest.raises(RuntimeError):
    getInterpolators(ds, kind='pointwise')
    assert 'Weather model contains NaNs!' in caplog.text, 'No warning was raised!'


def test_transformPoints():
    lats = np.array([10, 20, 30, 45, 75, 80, 90])
    lons = np.array([0,-180,180, 90,-90,-20, 10])
    hts  = np.array([0,   0,  0,  0,  0,  0,  0])

    epsg4326 = CRS.from_epsg(4326)
    ecef = CRS.from_epsg(4978)

    out = transformPoints(lats, lons, hts, epsg4326, ecef)
    y,x,z=out[:,0], out[:,1], out[:,2]

    T = Transformer.from_crs(4978, 4326)

    test = T.transform(x,y,z)
    assert np.allclose(test[0], lats)
    assert np.allclose(test[1], lons)
    assert np.allclose(test[2], hts)


def test_transformPoints_2(hrrr_proj):
    hrrr_proj = hrrr_proj
    lats = np.array([40, 45, 55])
    lons = np.array([-90,-90, -90])
    hts  = np.array([0,   0,  0])

    epsg4326 = CRS.from_epsg(4326)

    out = transformPoints(lats, lons, hts, epsg4326, hrrr_proj)
    y,x,z=out[:,0], out[:,1], out[:,2]

    T = Transformer.from_crs(hrrr_proj, 4326)

    test = T.transform(x,y,z)
    assert np.allclose(test[0], lats)
    assert np.allclose(test[1], lons)
    assert np.allclose(test[2], hts)


def test_transformPoints_3():
    lats = np.array([0, 0, 0])
    lons = np.array([0,-90, 180])
    hts  = np.array([0,   0,  0])

    epsg4326 = CRS.from_epsg(4326)
    ecef = CRS.from_epsg(4978)

    out = transformPoints(lats, lons, hts, epsg4326, ecef)
    y,x,z=out[:,0], out[:,1], out[:,2]

    assert np.allclose(x, [6378137., 0, -6378137])
    assert np.allclose(y, [0, -6378137., 0])
    assert np.allclose(z, [0, 0, 0])
    