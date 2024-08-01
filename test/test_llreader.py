import os
from pathlib import Path
import pytest

import numpy as np
import pandas as pd

from test import GEOM_DIR, TEST_DIR
from pyproj import CRS

from RAiDER.cli.raider import calcDelays

from RAiDER.utilFcns import rio_open
from RAiDER.llreader import (
    StationFile, RasterRDR, BoundingBox, GeocodedFile, bounds_from_latlon_rasters, bounds_from_csv
)

SCENARIO0_DIR = TEST_DIR / "scenario_0"
SCENARIO1_DIR = TEST_DIR / "scenario_1/geom"
SCENARIO2_DIR = TEST_DIR / "scenario_2"


@pytest.fixture
def parser():
    return calcDelays()


@pytest.fixture
def station_file():
    return SCENARIO2_DIR / 'stations.csv'


@pytest.fixture
def llfiles():
    return SCENARIO1_DIR / 'lat.dat', SCENARIO1_DIR / 'lon.dat'


def test_latlon_reader_2():
    with pytest.raises(ValueError):
        RasterRDR(lat_file=None, lon_file=None)

    with pytest.raises(ValueError):
        RasterRDR(lat_file='doesnotexist.rdr', lon_file='doesnotexist.rdr')


def test_aoi_epsg():
    bbox = [20, 27, -115, -104]
    r = BoundingBox(bbox)
    r.set_output_spacing(ll_res=0.05)
    test = r.get_output_spacing(4978)
    assert test == 0.05 * 1e5


def test_set_output_dir():
    bbox = [20, 27, -115, -104]
    r = BoundingBox(bbox)
    r.set_output_directory('dummy_directory')
    assert r._output_directory == 'dummy_directory'


def test_set_xygrid():
    bbox = [20, 27, -115, -104]
    crs = CRS.from_epsg(4326)
    r = BoundingBox(bbox)
    r.set_output_spacing(ll_res=0.1)
    r.set_output_xygrid(dst_crs=4978)
    r.set_output_xygrid(dst_crs=crs)
    assert True


def test_latlon_reader():
    latfile = Path(GEOM_DIR) / 'lat.rdr'
    lonfile = Path(GEOM_DIR) / 'lon.rdr'
    lat_true = rio_open(latfile)
    lon_true = rio_open(lonfile)

    query = RasterRDR(lat_file=str(latfile), lon_file=str(lonfile))
    lats, lons = query.readLL()
    assert lats.shape == (45, 226)
    assert lons.shape == (45, 226)

    assert np.allclose(lat_true, lats, equal_nan=True)
    assert np.allclose(lon_true, lons, equal_nan=True)

    # Hard code the lat/lon bounds to test against changing the files
    bounds_true = [15.7637, 21.4936, -101.6384, -98.2418]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(query.bounds(), bounds_true)])


def test_badllfiles(station_file):
    latfile = os.path.join(GEOM_DIR, 'lat.rdr')
    lonfile = os.path.join(GEOM_DIR, 'lon_dummy.rdr')
    station_file = station_file
    with pytest.raises(ValueError):
        RasterRDR(lat_file=latfile, lon_file=lonfile)
    with pytest.raises(ValueError):
        RasterRDR(lat_file=latfile, lon_file=station_file)
    with pytest.raises(ValueError):
        RasterRDR(lat_file=station_file, lon_file=lonfile)

def test_read_bbox():
    bbox = [20, 27, -115, -104]
    query = BoundingBox(bbox)
    assert query.type() == 'bounding_box'
    assert query.bounds() == bbox
    assert query.projection() == 'EPSG:4326'


def test_read_station_file(station_file):
    query = StationFile(station_file)
    lats, lons = query.readLL()
    stats = pd.read_csv(station_file).drop_duplicates(subset=["Lat", "Lon"])

    assert np.allclose(lats, stats['Lat'].values)
    assert np.allclose(lons, stats['Lon'].values)

    assert query.projection() == 'EPSG:4326'

    # Hard code the lat/lon bounds to test against changing the files
    bounds_true = [33.746, 36.795, -118.312, -114.892]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(query.bounds(), bounds_true)])


def test_bounds_from_latlon_rasters():
    lat_path = Path(GEOM_DIR) / 'lat.rdr'
    lon_path = Path(GEOM_DIR) / 'lon.rdr'
    snwe, _, _ = bounds_from_latlon_rasters(str(lat_path), str(lon_path))

    bounds_true =[15.7637, 21.4936, -101.6384, -98.2418]
    assert all([np.allclose(b, t, rtol=1e-4) for b, t in zip(snwe, bounds_true)])


def test_bounds_from_csv(station_file):
    bounds_true = [33.746, 36.795, -118.312, -114.892]
    snwe = bounds_from_csv(station_file)
    assert all([np.allclose(b, t) for b, t in zip(snwe, bounds_true)])


def test_readZ_sf(station_file):
    aoi = StationFile(station_file)
    assert np.allclose(aoi.readZ(), .1)


def test_GeocodedFile():
    aoi = GeocodedFile(SCENARIO0_DIR / 'small_dem.tif', is_dem=True)
    z = aoi.readZ()
    x,y = aoi.readLL()
    assert z.shape == (569,558)
    assert x.shape == z.shape
