import os
import pytest

import numpy as np
import pandas as pd

from argparse import ArgumentParser
from test import GEOM_DIR, TEST_DIR

import RAiDER.runProgram

from RAiDER.ioFcns import gdal_open
from RAiDER.llreader import (
    readLLFromLLFiles, readLLFromBBox, readLLFromStationFile, forceNDArray
)

SCENARIO2_DIR = os.path.join(TEST_DIR, "scenario_2")
SCENARIO1_DIR = os.path.join(TEST_DIR, "scenario_1", "geom")


@pytest.fixture
def parser():
    return RAiDER.runProgram.create_parser()


@pytest.fixture
def station_file():
    return os.path.join(SCENARIO2_DIR, 'stations.csv')


@pytest.fixture
def llfiles():
    return os.path.join(SCENARIO1_DIR, 'lat.dat'), os.path.join(SCENARIO1_DIR, 'lon.dat')


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


def test_stationfile_reader(station_file):
    lats, lons, llproj = readLLFromStationFile(station_file)
    assert len(lats) == 8


def test_forceNDArray():
    assert np.all(np.array([1, 2, 3]) == forceNDArray([1, 2, 3]))
    assert np.all(np.array([1, 2, 3]) == forceNDArray((1, 2, 3)))
    assert forceNDArray(None) is None


def test_readLL_bbox(parser):
    args = parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        "--bbox", "20", "27", "-115", "-104",
    ])
    bbox = [20, 27, -115, -104]

    assert args.query_area == (bbox)

    lats, lons, proj = readLLFromBBox(args.query_area)

    assert np.allclose(lats, np.array([20, 27]))
    assert np.allclose(lons, np.array([-115, -104]))
    assert proj == 'EPSG:4326'


def test_readLL_file(parser, station_file):
    args = parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        "--station_file", station_file
    ])

    assert args.query_area == (station_file)

    lats, lons, proj = readLLFromStationFile(args.query_area)
    stats = pd.read_csv(station_file)

    assert np.allclose(lats, stats['Lat'].values)
    assert np.allclose(lons, stats['Lon'].values)
    assert proj == 'EPSG:4326'


def test_readLL_files(parser, llfiles):
    latfile, lonfile = llfiles
    args = parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        "--latlon", latfile, lonfile,
    ])
    assert args.query_area == ([latfile, lonfile])

    lat_true = gdal_open(latfile)
    lon_true = gdal_open(lonfile)
    lats, lons, proj = readLLFromLLFiles(*args.query_area)

    assert np.allclose(lat_true, lats)
    assert np.allclose(lon_true, lons)
    assert proj == ''
