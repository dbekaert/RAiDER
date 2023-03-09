import datetime
import h5py
import os
import pytest

import numpy as np
import pyproj
import rasterio

from test import TEST_DIR

from RAiDER.utilFcns import (
    _least_nonzero, cosd, rio_open, sind,
    writeArrayToRaster, rio_profile,
    rio_extents, getTimeFromFile, enu2ecef, ecef2enu,
    transform_bbox, clip_bbox, get_nearest_wmtimes,
)


_R_EARTH = 6378138

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")


@pytest.fixture
def make_points_0d_data():
    return (
        np.stack([
            np.zeros(200),
            np.zeros(200),
            np.arange(0, 1000, 5)
        ],
            axis=-1
        ).T,
        (1000., np.array([0., 0., 0.]), np.array([0., 0., 1.]), 5.)
    )


@pytest.fixture
def make_points_1d_data():
    ray1 = np.stack([
        np.zeros(200),
        np.zeros(200),
        np.arange(0, 1000, 5)
    ],
        axis=-1
    ).T
    ray2 = np.stack([
        np.zeros(200),
        np.arange(0, 1000, 5),
        np.zeros(200),
    ],
        axis=-1
    ).T
    rays = np.stack([ray1, ray2], axis=0)

    sp = np.array([[0., 0., 0.],
                   [0., 0., 0.]])
    slv = np.array([[0., 0., 1.],
                    [0., 1., 0.]])
    return rays, (1000., sp, slv, 5.)


@pytest.fixture
def make_points_2d_data():
    sp = np.zeros((2, 2, 3))
    slv = np.zeros((2, 2, 3))
    slv[0, 0, 0] = 1
    slv[0, 1, 1] = 1
    slv[1, 0, 2] = 1
    slv[1, 1, 0] = -1
    make_points_args = (20., sp, slv, 5)

    rays = np.array([[[[0., 5., 10., 15.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]],
                      [[0., 0., 0., 0.],
                       [0., 5., 10., 15.],
                       [0., 0., 0., 0.]]],
                     [[[0., 0., 0., 0.],
                       [0., 0., 0., 0.],
                       [0., 5., 10., 15.]],
                      [[0., -5., -10., -15.],
                       [0., 0., 0., 0.],
                       [0., 0., 0., 0.]]]])

    return rays, make_points_args


@pytest.fixture
def make_points_3d_data():
    sp = np.zeros((3, 3, 3, 3))
    sp[:, :, 1, 2] = 10
    sp[:, :, 2, 2] = 100
    slv = np.zeros((3, 3, 3, 3))
    slv[0, :, :, 2] = 1
    slv[1, :, :, 1] = 1
    slv[2, :, :, 0] = 1

    make_points_args = (100., sp, slv, 5)

    df = np.loadtxt(os.path.join(TEST_DIR, "test_result_makePoints3D.txt"))

    return df.reshape((3, 3, 3, 3, 20)), make_points_args


def test_sind():
    assert np.allclose(
        sind(np.array([0, 30, 90, 180])),
        np.array([0, 0.5, 1, 0])
    )


def test_cosd():
    assert np.allclose(
        cosd(np.array([0, 60, 90, 180])),
        np.array([1, 0.5, 0, -1])
    )


def test_rio_open():
    out = rio_open(os.path.join(TEST_DIR, "test_geom", "lat.rdr"), False)

    assert np.allclose(out.shape, (45, 226))


def test_writeArrayToRaster(tmp_path):
    array = np.transpose(
        np.array([np.arange(0, 10)])
    ) * np.arange(0, 10)
    filename = str(tmp_path / 'dummy.out')

    writeArrayToRaster(array, filename)
    with rasterio.open(filename) as src:
        band = src.read(1)
        noval = src.nodatavals[0]


    assert noval == 0
    assert np.allclose(band, array)


def test_makePoints0D_cython(make_points_0d_data):
    from RAiDER.makePoints import makePoints0D

    true_ray, args = make_points_0d_data

    test_result = makePoints0D(*args)
    assert np.allclose(test_result, true_ray)


def test_makePoints1D_cython(make_points_1d_data):
    from RAiDER.makePoints import makePoints1D

    true_ray, args = make_points_1d_data

    test_result = makePoints1D(*args)
    assert np.allclose(test_result, true_ray)


def test_makePoints2D_cython(make_points_2d_data):
    from RAiDER.makePoints import makePoints2D

    true_ray, args = make_points_2d_data

    test_result = makePoints2D(*args)
    assert np.allclose(test_result, true_ray)


def test_makePoints3D_Cython_values(make_points_3d_data):
    from RAiDER.makePoints import makePoints3D

    true_rays, args = make_points_3d_data

    test_result = makePoints3D(*args)

    assert test_result.ndim == 5
    assert np.allclose(test_result, true_rays)


def test_least_nonzero():
    a = np.arange(20, dtype="float64").reshape(2, 2, 5)
    a[0, 0, 0] = np.nan
    a[1, 1, 0] = np.nan

    assert np.allclose(
        _least_nonzero(a),
        np.array([
            [1, 5],
            [10, 16]
        ]),
        atol=1e-16
    )


def test_least_nonzero_2():
    a = np.array([
        [[10., 5., np.nan],
         [11., np.nan, 1],
         [18, 17, 16]],

        [[np.nan, 12., 6.],
         [np.nan, 13., 20.],
         [np.nan, np.nan, np.nan]]
    ])

    assert np.allclose(
        _least_nonzero(a),
        np.array([
            [10, 11, 18],
            [12, 13, np.nan]
        ]),
        atol=1e-16,
        equal_nan=True
    )


def test_rio_extent():
    # Create a simple georeferenced test file
    with rasterio.open("test.tif", mode="w",
                       width=11, height=11, count=1,
                       dtype=np.float64, crs=pyproj.CRS.from_epsg(4326),
                       transform=rasterio.Affine.from_gdal(
                           17.0, 0.1, 0, 18.0, 0, -0.1
                       )) as dst:
        dst.write(np.random.randn(11, 11), 1)
    profile = rio_profile("test.tif")
    assert rio_extents(profile) == (17.0, 18.0, 17.0, 18.0)
    os.remove("test.tif")


def test_rio_extent2():
    with pytest.raises(AttributeError):
        rio_profile(os.path.join(TEST_DIR, "test_geom", "lat.rdr"))


def test_getTimeFromFile():
    name1 = 'abcd_2020_01_01_T00_00_00jijk.xyz'
    assert getTimeFromFile(name1) == datetime.datetime(2020, 1, 1, 0, 0, 0)


def test_project():
    #   the true UTM coordinates are extracted from this website as an independent check: https://www.latlong.net/lat-long-utm.html
    from RAiDER.utilFcns import project
    #   Hawaii
    true_utm = (5, 'Q', 212721.65, 2192571.64)
    tup = project((-155.742188, 19.808054))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   New Zealand
    true_utm = (59, 'G', 645808.07, 5373216.94)
    tup = project((172.754517, -41.779505))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   UK
    true_utm = (30, 'U', 693205.98, 5742711.01)
    tup = project((-0.197754, 51.801822))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   US
    true_utm = (14, 'S', 640925.54, 4267877.48)
    tup = project((-97.382813, 38.548165))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   China
    true_utm = (48, 'S', 738881.72, 3734577.12)
    tup = project((107.578125, 33.724340))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   South Africa
    true_utm = (34, 'J', 713817.66, 6747653.92)
    tup = project((23.203125, -29.382175))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   Argintina
    true_utm = (19, 'H', 628210.60, 5581184.24)
    tup = project((-67.500000, -39.909736))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]

    #   Greenland
    true_utm = (24, 'X', 475105.61, 8665516.77)
    tup = project((-40.078125, 78.061989))
    assert np.allclose((tup[0], tup[2], tup[3]), (true_utm[0], true_utm[2], true_utm[3]))
    assert tup[1] == true_utm[1]


def test_WGS84_to_UTM():
    from RAiDER.utilFcns import WGS84_to_UTM

    lats = np.array([38.0, 38.0, 38.0])
    lons = np.array([-97.0, -92.0, -87.0])

    # true utm coodinates at local zones (14, 15, 16)
    true_utm_local = np.array([[14, 675603.37, 4207702.37], [15, 587798.42, 4206286.76], [16, 500000.00, 4205815.02]])
    true_utm_local_letter = np.array(['S', 'S', 'S'])

    # true utm coordinates at the zone of the center (15)
    # created using the following line
    # pyproj.Proj(proj='utm', zone=15, ellps='WGS84')(lons,lats)
    true_utm_common = np.array([[15, 148741.08527017, 4213370.735271454], [15, 587798.42, 4206286.76], [15, 1027018.2271954522, 4222839.127299805]])
    true_utm_common_letter = np.array(['S', 'S', 'S'])

    #   use local UTM zones
    Z, L, X, Y = WGS84_to_UTM(lons, lats)
    cal_utm_local = np.array([Z, X, Y]).transpose()
    assert np.allclose(true_utm_local, cal_utm_local)
    assert np.all(true_utm_local_letter == L)

    #   use common UTM zone
    Z, L, X, Y = WGS84_to_UTM(lons, lats, common_center=True)
    cal_utm_common = np.array([Z, X, Y]).transpose()
    assert np.allclose(true_utm_common, cal_utm_common)
    assert np.all(true_utm_common_letter == L)


@pytest.mark.skipif(True, reason='Need to ensure this file always get written before this executes')
def test_read_weather_model_file():
    # TODO: read_wm_file is undefined
    weather_model_obj = read_wm_file(
        os.path.join(
            SCENARIO_DIR,
            'weather_files',
            'ERA5_2020_01_03_T23_00_00_15.75N_18.25N_103.24W_99.75W.nc'
        )
    )
    assert weather_model_obj.Model() == 'ERA-5'


def test_enu2ecef_1():
    enu = np.array([0, 0, 1])
    llh = np.array([0, 0, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([1, 0, 0]))


def test_enu2ecef_2():
    enu = np.array([0, 0, 1])
    llh = np.array([0, 90, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 1, 0]))


def test_enu2ecef_3():
    enu = np.array([0, 0, 1])
    llh = np.array([0, -90, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, -1, 0]))


def test_enu2ecef_4():
    enu = np.array([0, 0, 1])
    llh = np.array([90, 0, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 0, 1]))


def test_enu2ecef_5():
    enu = np.array([0, 0, 1])
    llh = np.array([-90, 0, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 0, -1]))


def test_enu2ecef_6():
    enu = np.array([0, 1, 0])
    llh = np.array([0, 0, 0])
    ecef = enu2ecef(enu[0], enu[1], enu[2], llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 0, 1]))


def test_ecef2enu_1():
    enu = np.array([0, 0, 1])
    llh = np.array([0, 0, 0])
    enu = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(enu, np.array([0, 1, 0]))


def test_ecef2enu_2():
    enu = np.array([0, 0, 1])
    llh = np.array([0, 90, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 1, 0]))


def test_ecef2enu_3():
    enu = np.array([0, 0, 1])
    llh = np.array([0, -90, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 1, 0]))


def test_ecef2enu_4():
    enu = np.array([0, 0, 1])
    llh = np.array([90, 0, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 0, 1]))


def test_ecef2enu_5():
    enu = np.array([0, 0, 1])
    llh = np.array([-90, 0, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 0, -1]))


def test_ecef2enu_6():
    enu = np.array([0, 0, -1])
    llh = np.array([0, -180, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, -1, 0]))


def test_ecef2enu_7():
    enu = np.array([0, 0, 1])
    llh = np.array([0, -180, 1000])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([0, 1, 0]))


def test_ecef2enu_8():
    enu = np.array([1, 1, 0])
    llh = np.array([0, 0, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([1, 0, 1]))


def test_ecef2enu_9():
    enu = np.array([1, 1, 0])
    llh = np.array([0, 180, 0])
    ecef = ecef2enu(enu, llh[0], llh[1], llh[2])
    assert np.allclose(ecef, np.array([-1, 0, -1]))


def test_transform_bbox_1():
    wesn = [-77.0, -76.0, 34.0, 35.0]
    snwe = wesn[2:] + wesn[:2]

    assert transform_bbox(snwe, src_crs=4326, dest_crs=4326) == snwe


def test_transform_bbox_2():
    snwe_in = [34.0, 35.0, -77.0, -76.0]

    expected_snwe = [3762606.6598762725,
                     3874870.6347308,
                     315290.16886786406,
                     408746.7471660769]

    snwe = transform_bbox(snwe_in, src_crs=4326, dest_crs=32618, margin=0.)
    assert np.allclose(snwe, expected_snwe)

def test_clip_bbox():
    wesn = [-77.0, -76.0, 34.0, 35.0]
    snwe = [34.0, 35.01, -77.0, -76.0]
    snwe_in = [34.005, 35.0006, -76.999, -76.0]
    assert clip_bbox(wesn, 0.01) == wesn
    assert clip_bbox(snwe_in, 0.01) == snwe

def test_get_nearest_wmtimes():
    t0 = datetime.datetime(2020,1,1,11,35,0)
    test_out = get_nearest_wmtimes(t0, 3)
    true_out = [datetime.datetime(2020, 1, 1, 9, 0), datetime.datetime(2020, 1, 1, 12, 0)]
    assert [t == t0 for t, t0 in zip(test_out, true_out)]

def test_get_nearest_wmtimes_2():
    t0 = datetime.datetime(2020,1,1,11,3,0)
    test_out = get_nearest_wmtimes(t0, 1)
    true_out = [datetime.datetime(2020, 1, 1, 11, 0)]
    assert [t == t0 for t, t0 in zip(test_out, true_out)]

def test_get_nearest_wmtimes_3():
    t0 = datetime.datetime(2020,1,1,11,57,0)
    test_out = get_nearest_wmtimes(t0, 3)
    true_out = [datetime.datetime(2020, 1, 1, 12, 0)]
    assert [t == t0 for t, t0 in zip(test_out, true_out)]

def test_get_nearest_wmtimes_4():
    t0 = datetime.datetime(2020,1,1,11,25,0)
    test_out = get_nearest_wmtimes(t0, 1)
    true_out = [datetime.datetime(2020, 1, 1, 11, 0), datetime.datetime(2020, 1, 1, 12, 0)]
    assert [t == t0 for t, t0 in zip(test_out, true_out)]
