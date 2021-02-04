import datetime
import h5py
import os
import pytest

import numpy as np

from datetime import time
from test import TEST_DIR
from osgeo import gdal, osr

from RAiDER.utilFcns import (
    _least_nonzero, cosd, gdal_open, makeDelayFileNames, sind,
    writeArrayToRaster, writeResultsToHDF5, gdal_extents, modelName2Module,
    getTimeFromFile
)


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


def test_gdal_open():
    out = gdal_open(os.path.join(TEST_DIR, "test_geom", "lat.rdr"), False)

    assert np.allclose(out.shape, (45, 226))


def test_writeResultsToHDF5(tmp_path):
    lats = np.array([15.0, 15.5, 16.0, 16.5, 17.5, -40, 60, 90])
    lons = np.array([-100.0, -100.4, -91.2, 45.0, 0., -100, -100, -100])
    hgts = np.array([0., 1000., 10000., 0., 0., 0., 0., 0.])
    wet = np.zeros(lats.shape)
    hydro = np.ones(lats.shape)
    filename = str(tmp_path / 'dummy.hdf5')

    writeResultsToHDF5(lats, lons, hgts, wet, hydro, filename)

    with h5py.File(filename, 'r') as f:
        assert np.allclose(np.array(f['lat']), lats)
        assert np.allclose(np.array(f['hydroDelay']), hydro)
        assert f.attrs['DelayType'] == 'Zenith'


def test_writeArrayToRaster(tmp_path):
    array = np.transpose(
        np.array([np.arange(0, 10)])
    ) * np.arange(0, 10)
    filename = str(tmp_path / 'dummy.out')

    writeArrayToRaster(array, filename)
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    band = dataset.GetRasterBand(1)

    assert band.GetNoDataValue() == 0
    assert np.allclose(band.ReadAsArray(), array)


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


def test_makeDelayFileNames():
    assert makeDelayFileNames(None, None, "h5", "name", "dir") == \
        ("dir/name_wet_ztd.h5", "dir/name_hydro_ztd.h5")

    assert makeDelayFileNames(None, (), "h5", "name", "dir") == \
        ("dir/name_wet_std.h5", "dir/name_hydro_std.h5")

    assert makeDelayFileNames(time(1, 2, 3), None, "h5", "model_name", "dir") == \
        (
            "dir/model_name_wet_01_02_03_ztd.h5",
            "dir/model_name_hydro_01_02_03_ztd.h5"
    )

    assert makeDelayFileNames(time(1, 2, 3), "los", "h5", "model_name", "dir") == \
        (
            "dir/model_name_wet_01_02_03_std.h5",
            "dir/model_name_hydro_01_02_03_std.h5"
    )


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


def test_gdal_extent():
    # Create a simple georeferenced test file
    ds = gdal.GetDriverByName('GTiff').Create('test.tif', 11, 11, 1, gdal.GDT_Float64)
    ds.SetGeoTransform((17.0, 0.1, 0, 18.0, 0, -0.1))
    band = ds.GetRasterBand(1)
    band.WriteArray(np.random.randn(11, 11))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    ds.SetProjection(srs.ExportToWkt())
    ds = None
    band = None

    assert gdal_extents('test.tif') == [17.0, 18.0, 18.0, 17.0]


def test_gdal_extent2():
    with pytest.raises(AttributeError):
        gdal_extents(os.path.join(TEST_DIR, "test_geom", "lat.rdr"))


def test_getTimeFromFile():
    name1 = 'abcd_2020_01_01_T00_00_00jijk.xyz'
    assert getTimeFromFile(name1) == datetime.datetime(2020, 1, 1, 0, 0, 0)


def test_model2module():
    model_module_name, model_obj = modelName2Module('ERA5')
    assert model_obj().Model() == 'ERA-5'


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
