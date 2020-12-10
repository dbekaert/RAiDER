import os
import pyproj

import numpy as np

from RAiDER.geometry import (
    project,
    unproject,
    WGS84_to_UTM
)

def test_project():
    # The true UTM coordinates are extracted from 
    # https://www.latlong.net/lat-long-utm.html
    # as an independent check

    #   Hawaii
    true_utm = (5, 'Q', 212721.65, 2192571.64)
    tup = project((-155.742188,19.808054))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   New Zealand
    true_utm = (59, 'G', 645808.07, 5373216.94)
    tup = project((172.754517,-41.779505))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   UK
    true_utm = (30, 'U', 693205.98, 5742711.01)
    tup = project((-0.197754,51.801822))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]

    #   US
    true_utm = (14, 'S', 640925.54, 4267877.48)
    tup = project((-97.382813,38.548165))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   China
    true_utm = (48, 'S', 738881.72, 3734577.12)
    tup = project((107.578125,33.724340))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   South Africa
    true_utm = (34, 'J', 713817.66, 6747653.92)
    tup = project((23.203125,-29.382175))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   Argentina
    true_utm = (19, 'H', 628210.60, 5581184.24)
    tup = project((-67.500000,-39.909736))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]
    
    #   Greenland
    true_utm = (24, 'X', 475105.61, 8665516.77)
    tup = project((-40.078125,78.061989))
    assert np.allclose(
            (tup[0],tup[2],tup[3]),
            (true_utm[0],true_utm[2],true_utm[3])
        )
    assert tup[1] == true_utm[1]


def test_WGS84_to_UTM():
    lats = np.array([38.0, 38.0, 38.0])
    lons = np.array([-97.0, -92.0, -87.0])
    
    # true utm coodinates at local zones (14, 15, 16)
    true_utm_local = np.array(
            [
                [14, 675603.37, 4207702.37],
                [15, 587798.42, 4206286.76],
                [16, 500000.00, 4205815.02]
            ]
        )
    true_utm_local_letter = np.array(['S','S','S'])
    
    # true utm coordinates at the zone of the center (15)
    # created using the following line
    # pyproj.Proj(proj='utm', zone=15, ellps='WGS84')(lons,lats)
    true_utm_common = np.array(
            [
                [15, 148741.08527017, 4213370.735271454],
                [15, 587798.42, 4206286.76],
                [15, 1027018.2271954522, 4222839.127299805]
            ]
        )
    true_utm_common_letter = np.array(['S','S','S'])
    
    #   use local UTM zones
    Z, L, X, Y = WGS84_to_UTM(lons, lats)
    cal_utm_local = np.array([Z,X,Y]).transpose()
    assert np.allclose(true_utm_local, cal_utm_local)
    assert np.all(true_utm_local_letter == L)
    
    #   use common UTM zone
    Z, L, X, Y = WGS84_to_UTM(lons, lats, common_center=True)
    cal_utm_common = np.array([Z,X,Y]).transpose()
    assert np.allclose(true_utm_common, cal_utm_common)
    assert np.all(true_utm_common_letter == L)
