import datetime
import os
import pytest
from test import DATA_DIR

import numpy as np

from RAiDER.losreader import (
    read_ESA_Orbit_file,
    read_txt_file,
    cut_times,
    los_to_lv,
    get_sv,
)

@pytest.fixture
def svs():
    ref_time = datetime.datetime(2018, 11, 12, 23, 0, 42)
    T = [
        datetime.datetime(2018, 11, 12, 23, 0, 2),
        datetime.datetime(2018, 11, 12, 23, 0, 12),
        datetime.datetime(2018, 11, 12, 23, 0, 22),
        datetime.datetime(2018, 11, 12, 23, 0, 32),
        datetime.datetime(2018, 11, 12, 23, 0, 42),
        datetime.datetime(2018, 11, 12, 23, 0, 52),
        datetime.datetime(2018, 11, 12, 23, 1, 2),
        datetime.datetime(2018, 11, 12, 23, 1, 12),
    ]
    tr = np.array([(t - ref_time).total_seconds() for t in T])
    x = np.array([
        -2064965.285362,
	-2056228.553736,
	-2047224.526705,
	-2037955.293282,
	-2028422.977002,
	-2018629.735564,
	-2008577.760461,
	-1998269.276601,
    ])
    y = np.array([
        6434865.494987,
	6460407.492520,
	6485212.031660,
	6509275.946120,
	6532596.156540,
	6555169.670917,
	6576993.585012,
	6598065.082739,
    ])
    z = np.array([
	2090670.967443,
	2019650.417312,
	1948401.684024,
	1876932.818066,
	1805251.894958,
	1733367.014327,
	1661286.298987,
	1589017.893976,
    ])
    vx = np.array([
	860.239634,
	887.072466,
	913.698134,
	940.113169,
	966.314136,
	992.297636,
	1018.060311,
	1043.598837,
    ])
    vy = np.array([
        2590.964968,
	2517.380329,
	2443.474728,
	2369.256838,
	2294.735374,
	2219.919093,
	2144.816789,
	2069.437298,
    ])
    vz = np.array([
	-7090.378144,
	-7113.598127,
	-7136.014344,
	-7157.624244,
	-7178.425371,
	-7198.415359,
	-7217.591940,
	-7235.952940,
    ])
    return [tr, x, y, z, vx, vy, vz], ref_time


def test_read_ESA_Orbit_file(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'S1_orbit_example.EOF')
    svs = read_ESA_Orbit_file(filename, ref_time)    
    assert [np.allclose(s, ts) for s, ts in zip(svs, true_svs)]

def test_read_txt_file(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'S1_sv_file.txt')
    svs = read_txt_file(filename)    
    assert [np.allclose(s, ts) for s, ts in zip(svs, true_svs)]

def test_get_sv_1(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'S1_orbit_example.EOF')
    svs = get_sv(filename, ref_time)
    assert [np.allclose(s, ts) for s, ts in zip(svs, true_svs)]
    
def test_get_sv_2(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'S1_sv_file.txt')
    svs = get_sv(filename, ref_time)
    assert [np.allclose(s, ts) for s, ts in zip(svs, true_svs)]
    
def test_get_sv_3(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'geom', 'warpedDEM.dem')
    with pytest.raises(ValueError):
        get_sv(filename, ref_time)
    
def test_get_sv_4(svs):
    true_svs, ref_time = svs
    filename = os.path.join(DATA_DIR, 'no_exist.txt')
    with pytest.raises(FileNotFoundError):
        get_sv(filename, ref_time)
    
def test_cut_times(svs):
    true_svs, ref_time = svs
    assert len(true_svs[0][cut_times(true_svs[0])]) == len(true_svs[0])

def test_cut_times_2(svs):
    true_svs, ref_time = svs
    assert len(true_svs[0][cut_times(true_svs[0], pad=5)]) == 1

def test_cut_times_3(svs):
    true_svs, ref_time = svs
    assert len(true_svs[0][cut_times(true_svs[0], pad=15)]) == 3

def test_cut_times_4(svs):
    true_svs, ref_time = svs
    
    assert len(true_svs[0][cut_times(true_svs[0], pad=400)]) == len(true_svs[0])

def test_los_to_lv():
    with pytest.raises(ValueError):
        los_to_lv(-10, 0)

def test_los_to_lv_2():
    assert np.allclose(
        los_to_lv(0, 0), 
        np.array([0, 0, 1]) 
   )

def test_los_to_lv_3():
    assert np.allclose(
        los_to_lv(0, -180), 
        np.array([0, 0, 1]) 
   )

def test_los_to_lv_3b():
    assert np.allclose(
        los_to_lv(0, 18), 
        np.array([0, 0, 1]) 
   )

def test_los_to_lv_3c():
    assert np.allclose(
        los_to_lv(0, -18), 
        np.array([0, 0, 1]) 
   )

def test_los_to_lv_4():
    assert np.allclose(
        los_to_lv(35, 0), 
        np.array([0, np.sin(np.radians(35)), np.cos(np.radians(35))])
   )
    
def test_los_to_lv_5():
    assert np.allclose(
        los_to_lv(35, 180), 
        np.array([0, -np.sin(np.radians(35)), np.cos(np.radians(35))])
   )

def test_los_to_lv_6():
    assert np.allclose(
        los_to_lv(35, 90), 
        np.array([-np.sin(np.radians(35)), 0, np.cos(np.radians(35))])
   )
