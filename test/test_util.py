from datetime import time
from test import TEST_DIR

import h5py
import numpy as np
import pytest
from osgeo import gdal

from RAiDER.makeRays import makeRays1D, makeRays3D
from RAiDER.utilFcns import (
    _least_nonzero, cosd, gdal_open, makeDelayFileNames, sind,
    writeArrayToRaster, writeResultsToHDF5
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
def make_rays_1d_data():
    ray1 = np.stack([
        np.full(200, 6378137.),
        np.zeros(200),
        np.arange(0, 1000, 5)
    ],
        axis=-1
    ).T
    ray2 = np.stack([
        np.full(200, 6378137.),
        np.arange(0, 1000, 5),
        np.zeros(200),
    ],
        axis=-1
    ).T
    rays = np.stack([ray1, ray2], axis=0)

    sp = np.array([[6378137., 0., 0.],
                   [6378137., 0., 0.]])
    slv = np.array([[0., 0., 1.],
                    [0., 1., 0.]])
    # Since our rays are pointing parallel to the surface, it will take them a
    # long time to reach any sort of altitude. So we use a small number for the
    # altitude cutoff.
    return rays, (.078, sp, slv, 5.)


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
def make_rays_3d_data():
    sp = np.zeros((3, 3, 3, 3))
    sp[:, :, 1, 2] = 10
    sp[:, :, 2, 2] = 100
    slv = np.zeros((3, 3, 3, 3))
    slv[0, :, :, 2] = 1
    slv[1, :, :, 1] = 1
    slv[2, :, :, 0] = 1

    make_points_args = (100., sp, slv, 5)

    df = np.loadtxt(TEST_DIR / "test_result_makeRays3D.txt")

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
    out = gdal_open(TEST_DIR / "test_geom" / "lat.rdr", False)

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


def test_makeRays1D_along_axis():
    rays = makeRays1D(
        200,
        np.array([[6378137., 0., 0.]]),
        np.array([[1., 0., 0.]]),
        100.
    )

    assert np.allclose(rays, np.array([
        [
            [6378137., 6378237., 6378337.],
            [0., 0., 0.],
            [0., 0., 0.]
        ]
    ]))

    rays = makeRays1D(
        200,
        np.array([[0., 6378137., 0.]]),
        np.array([[0., 1., 0.]]),
        100.
    )

    assert np.allclose(rays, np.array([
        [
            [0., 0., 0.],
            [6378137., 6378237., 6378337.],
            [0., 0., 0.]
        ]
    ]))

    rays = makeRays1D(
        200,
        np.array([[0., 0., 6356752.]]),
        np.array([[0., 0., 1.]]),
        100.
    )

    assert np.allclose(rays, np.array([
        [
            [0., 0., 0.],
            [0., 0., 0.],
            [6356752., 6356852., 6356952.]
        ]
    ]))


def test_makeRays1D_along_zenith():
    rays = makeRays1D(
        200,
        np.array([[3909067., -3909067., -3170373.]]),
        np.array([[0.61237244, -0.61237244, -0.49999999]]),
        20.
    )

    assert np.allclose(rays, np.array([
        [
            np.linspace(3909067., 3909189.47, num=11),
            np.linspace(-3909067., -3909190.23, num=11),
            np.linspace(-3170373., -3170473.74, num=11)
        ]
    ]))


def test_makeRays1D_cython(make_rays_1d_data):
    true_ray, args = make_rays_1d_data

    test_result = makeRays1D(*args)
    assert np.allclose(test_result, true_ray)


def test_makeRays3D_Cython_values(make_rays_3d_data):
    true_rays, args = make_rays_3d_data

    test_result = makeRays3D(*args)

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
