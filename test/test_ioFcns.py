import datetime
import h5py
import os
import pytest

import numpy as np

from datetime import time
from test import TEST_DIR
from osgeo import gdal, osr

from RAiDER.ioFcns import (
    gdal_open, 
    makeDelayFileNames,
    writeArrayToRaster, 
    gdal_extents, 
    getTimeFromFile,
    writeVars2HDF5,
)


@pytest.fixture
def test_vars():
    ''' Generate vars for file I/O tests '''
    lats = np.arange(10, 50, 0.1)
    lons = np.arange(-100, -60, 0.1)
    hgts = np.random.randn(*lats.shape)
    wet = np.zeros(lats.shape)
    hydro = np.ones(lats.shape)

    d = {
            'lats': {'data': lats, 'attrs': {'proj': 'wgs-84'}},
            'lons': {'data': lons, 'attrs': {'proj': 'wgs-84'}},
            'hgts': {'data': hgts},
            'wet':  {'data': wet},
            'hydro':{'data': hydro}
        }
        
    return d


def test_writeVars2HDF5_1(tmp_path, test_vars):
    d = test_vars
    filename = str(tmp_path / 'dummy.h5')

    writeVars2HDF5(d, filename)
    
    with h5py.File(filename, 'r') as f:
        assert len(f.keys()) == 5
        lats = f['lats'][:]
        assert np.allclose(d['lats']['data'], lats)


def test_writeVars2HDF5_2(tmp_path, test_vars):
    d = test_vars
    filename = str(tmp_path / 'dummy2.h5')
    attrs = {'DelayType': 'Zenith'}

    writeVars2HDF5(d, filename, attrs)
    
    with h5py.File(filename, 'r') as f:
        assert len(f.keys()) == 5
        assert f.attrs['DelayType'] == 'Zenith'


def test_writeVars2HDF5_3(tmp_path, test_vars):
    d = test_vars
    filename = str(tmp_path / 'dummy2.h5')
    NDV = 0.
    chunkSize = (10,)

    writeVars2HDF5(
            d, 
            filename, 
            chunkSize = chunkSize, 
            NoDataValue = NDV
        )
    
    with h5py.File(filename, 'r') as f:
        assert len(f.keys()) == 5
        assert f['lats'].chunks == chunkSize
        assert f['lats'].fillvalue == NDV


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


def test_makeDelayFileNames1():
    f1, f2 = makeDelayFileNames(
            None, 
            None, 
            "h5", 
            "name", 
            "dir"
        ) 
    assert f1 == "dir/name_wet_ztd.h5"
    assert f2 == "dir/name_hydro_ztd.h5"

def test_makeDelayFileNames2():
    f1, f2 = makeDelayFileNames(
            None, 
            (),
            "h5", 
            "name", 
            "dir"
        )
    assert f1 == "dir/name_wet_std.h5"
    assert f2 == "dir/name_hydro_std.h5"

def test_makeDelayFileNames3():
    f1, f2 = makeDelayFileNames(
            time(1, 2, 3), 
            None, 
            "h5", 
            "model_name", 
            "dir"
        )
    assert f1 == "dir/model_name_wet_01_02_03_ztd.h5"
    assert f2 == "dir/model_name_hydro_01_02_03_ztd.h5"

def test_makeDelayFileNames4():
    f1, f2 = makeDelayFileNames(
            time(1, 2, 3), 
            "los", 
            "h5", 
            "model_name", 
            "dir"
        )
    assert f1 == "dir/model_name_wet_01_02_03_std.h5"
    assert f2 == "dir/model_name_hydro_01_02_03_std.h5"


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
        gdal_extents(
                os.path.join(
                    TEST_DIR, 
                    "test_geom", 
                    "lat.rdr"
                )
            )


def test_getTimeFromFile():
    name1 = 'abcd_2020_01_01_T00_00_00jijk.xyz'
    d1 = getTimeFromFile(name1)
    assert d1 == datetime.datetime(2020, 1, 1, 0, 0, 0)

