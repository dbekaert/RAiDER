import datetime
import os
import pytest

import multiprocessing as mp
import numpy as np
import pandas as pd

from test import TEST_DIR, pushd

import RAiDER.runProgram

from RAiDER.checkArgs import checkArgs, makeDelayFileNames, modelName2Module
from RAiDER.constants import _ZREF
from RAiDER.losreader import Zenith


SCENARIO_1 = os.path.join(TEST_DIR, "scenario_1")
SCENARIO_2 = os.path.join(TEST_DIR, "scenario_2")


def isWriteable(dirpath):
    '''Test whether a directory is writeable'''
    try:
        filehandle = open(os.path.join(dirpath, 'tmp.txt'), 'w')
        filehandle.close()
        return True
    except IOError:
        return False


@pytest.fixture
def parsed_args(tmp_path):
    parser = RAiDER.runProgram.create_parser()
    args = parser.parse_args([
        '--date', '20200103',
        '--time', '23:00:00',
        # '--latlon', 'latfile.dat', 'lonfile.dat',
        '--bbox', '-1', '1', '-1', '1',
        '--model', 'ERA5',
        '--outformat', 'hdf5'
    ])
    return args, parser


def test_checkArgs_outfmt_1(parsed_args):
    '''Test that passing height levels with hdf5 outformat works'''
    args, p = parsed_args
    args.outformat = 'hdf5'
    args.heightlvs = [10, 100, 1000]
    checkArgs(args, p)
    assert True


def test_checkArgs_outfmt_2(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.heightlvs = [10, 100, 1000]
    args.outformat = 'envi'
    with pytest.raises(ValueError):
        checkArgs(args, p)


def test_checkArgs_outfmt_3(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.query_area = os.path.join(SCENARIO_2, 'stations.csv')
    argDict = checkArgs(args, p)
    assert argDict['flag'] == 'station_file'


def test_checkArgs_outfmt_4(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.query_area = [os.path.join(SCENARIO_1, 'geom', 'lat.dat'), os.path.join(SCENARIO_1, 'geom', 'lat.dat')]
    argDict = checkArgs(args, p)
    assert argDict['flag'] == 'files'


def test_checkArgs_outfmt_5(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.query_area = os.path.join(SCENARIO_2, 'stations.csv')
    argDict = checkArgs(args, p)
    assert pd.read_csv(argDict['wetFilenames'][0]).shape == (8, 4)


def test_checkArgs_outloc_1(parsed_args):
    '''Test that the default output and weather model directories are correct'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    out = argDict['out']
    wmLoc = argDict['wmLoc']
    assert os.path.abspath(out) == os.getcwd()
    assert os.path.abspath(wmLoc) == os.path.join(os.getcwd(), 'weather_files')


def test_checkArgs_outloc_2(parsed_args, tmp_path):
    '''Tests that the correct output location gets assigned when provided'''
    with pushd(tmp_path):
        args, p = parsed_args
        args.out = tmp_path
        argDict = checkArgs(args, p)
        out = argDict['out']
        assert out == tmp_path


def test_checkArgs_outloc_2b(parsed_args, tmp_path):
    ''' Tests that the weather model directory gets passed through by itself'''
    with pushd(tmp_path):
        args, p = parsed_args
        args.out = tmp_path
        args.wmLoc = 'weather_dir'
        argDict = checkArgs(args, p)
        assert argDict['wmLoc'] == 'weather_dir'


def test_checkArgs_outloc_3(parsed_args):
    '''Tests that the weather model directory gets created when needed'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert os.path.isdir(argDict['wmLoc'])


def test_checkArgs_outloc_4(parsed_args):
    '''Tests for creating writeable weather model directory'''
    args, p = parsed_args
    argDict = checkArgs(args, p)

    assert isWriteable(argDict['wmLoc'])


def test_ll_bounds_1(parsed_args):
    '''Tests that lats out of bounds raises error'''
    args, p = parsed_args
    args.query_area[0] = -91
    with pytest.raises(ValueError):
        checkArgs(args, p)


def test_ll_bounds_2(parsed_args):
    '''Tests that lats out of bounds raises error'''
    args, p = parsed_args
    args.query_area[1] = 91
    with pytest.raises(ValueError):
        checkArgs(args, p)


def test_los_1(parsed_args):
    '''Tests that lats out of bounds raises error'''
    args, p = parsed_args
    args.lineofsight = 'los.rdr'
    argDict = checkArgs(args, p)
    assert argDict['los'][0] == 'los'
    assert argDict['los'][1] == 'los.rdr'


def test_los_2(parsed_args):
    '''Tests that lats out of bounds raises error'''
    args, p = parsed_args
    args.statevectors = 'sv.txt'
    argDict = checkArgs(args, p)
    assert argDict['los'][0] == 'sv'
    assert argDict['los'][1] == 'sv.txt'


def test_los_3(parsed_args):
    '''Tests that lats out of bounds raises error'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['los'] == Zenith


def test_models_1a(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'ERA-5'
    assert argDict['weather_model']['name'] == 'era5'


def test_models_1b(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    args.model = 'HRRR'
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'HRRR'
    assert argDict['weather_model']['name'] == 'hrrr'


def test_models_1c(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    args.model = 'NCMR'
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'NCMR'
    assert argDict['weather_model']['name'] == 'ncmr'


def test_models_1d(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    args.model = 'era-5'
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'ERA-5'
    assert argDict['weather_model']['name'] == 'era5'


def test_models_1e(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    args.model = 'ERA-5'
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'ERA-5'
    assert argDict['weather_model']['name'] == 'era5'


def test_models_1f(parsed_args):
    '''Tests that the weather model gets passed through correctly'''
    args, p = parsed_args
    args.model = 'Era-5'
    argDict = checkArgs(args, p)
    assert argDict['weather_model']['type'].Model() == 'ERA-5'
    assert argDict['weather_model']['name'] == 'era5'


def test_models_2(parsed_args):
    '''Tests that unknown weather models get rejected'''
    args, p = parsed_args
    args.model = 'unknown'
    with pytest.raises(NotImplementedError):
        checkArgs(args, p)


def test_models_3a(parsed_args):
    '''Tests that WRF weather models requires files'''
    args, p = parsed_args
    args.model = 'WRF'
    with pytest.raises(RuntimeError):
        checkArgs(args, p)


def test_models_3b(parsed_args):
    '''Tests that HDF5 weather models requires files'''
    args, p = parsed_args
    args.model = 'HDF5'
    with pytest.raises(RuntimeError):
        checkArgs(args, p)


def test_models_3c(parsed_args):
    '''Tests that WRF weather models requires files'''
    args, p = parsed_args
    args.model = 'WRF'
    args.files = ['file1.wrf', 'file2.wrf']
    # argDict = checkArgs(args, p)
    # TODO
    assert True


def test_zref_1(parsed_args):
    '''tests that default zref gets generated'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['zref'] == _ZREF


def test_zref_2(parsed_args):
    '''tests that default zref gets generated'''
    ztest = 20000
    args, p = parsed_args
    args.zref = ztest
    argDict = checkArgs(args, p)
    assert argDict['zref'] == ztest


def test_parallel_1(parsed_args):
    '''tests that parallel options are handled correctly'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['parallel'] == 1


def test_parallel_2(parsed_args):
    '''tests that parallel options are handled correctly'''
    args, p = parsed_args
    args.parallel = 'all'
    argDict = checkArgs(args, p)
    assert argDict['parallel'] == mp.cpu_count()


def test_parallel_3(parsed_args):
    '''tests that parallel options are handled correctly'''
    args, p = parsed_args
    args.parallel = 2
    argDict = checkArgs(args, p)
    assert argDict['parallel'] == 2


def test_parallel_4(parsed_args):
    '''tests that parallel options are handled correctly'''
    args, p = parsed_args
    args.parallel = 2000
    argDict = checkArgs(args, p)
    assert argDict['parallel'] == mp.cpu_count()


def test_verbose_1(parsed_args):
    '''tests that verbose option is handled correctly'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert not argDict['verbose']


def test_verbose_2(parsed_args):
    '''tests that verbose option is handled correctly'''
    args, p = parsed_args
    args.verbose = True
    argDict = checkArgs(args, p)
    assert argDict['verbose']


def test_download_only_1(parsed_args):
    '''tests that the download-only option is handled correctly'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert not argDict['download_only']


def test_download_only_2(parsed_args):
    '''tests that the download-only option is handled correctly'''
    args, p = parsed_args
    args.download_only = True
    argDict = checkArgs(args, p)
    assert argDict['download_only']


def test_useWeatherNodes_1(parsed_args):
    '''tests that the correct flag gets passed'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['flag'] == 'bounding_box'  # default arguments use a bounding box


def test_filenames_1(parsed_args):
    '''tests that the correct filenames are generated'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert 'Delay' not in argDict['wetFilenames'][0]
    assert 'wet' in argDict['wetFilenames'][0]
    assert 'hydro' in argDict['hydroFilenames'][0]
    assert '20200103' in argDict['wetFilenames'][0]
    assert '20200103' in argDict['hydroFilenames'][0]
    assert len(argDict['hydroFilenames']) == 1


def test_filenames_2(parsed_args):
    '''tests that the correct filenames are generated'''
    args, p = parsed_args
    args.query_area = os.path.join(SCENARIO_2, 'stations.csv')
    argDict = checkArgs(args, p)
    assert 'Delay' in argDict['wetFilenames'][0]
    assert '20200103' in argDict['wetFilenames'][0]
    assert len(argDict['wetFilenames']) == 1


def test_makeDelayFileNames_1():
    assert makeDelayFileNames(None, None, "h5", "name", "dir") == \
        ("dir/name_wet_ztd.h5", "dir/name_hydro_ztd.h5")


def test_makeDelayFileNames_2():
    assert makeDelayFileNames(None, (), "h5", "name", "dir") == \
        ("dir/name_wet_std.h5", "dir/name_hydro_std.h5")


def test_makeDelayFileNames_3():
    assert makeDelayFileNames(datetime.datetime(2020, 1, 1, 1, 2, 3), None, "h5", "model_name", "dir") == \
        (
            "dir/model_name_wet_20200101T010203_ztd.h5",
            "dir/model_name_hydro_20200101T010203_ztd.h5"
    )


def test_makeDelayFileNames_4():
    assert makeDelayFileNames(datetime.datetime(1900, 12, 31, 1, 2, 3), "los", "h5", "model_name", "dir") == \
        (
            "dir/model_name_wet_19001231T010203_std.h5",
            "dir/model_name_hydro_19001231T010203_std.h5"
    )


def test_model2module():
    model_module_name, model_obj = modelName2Module('ERA5')
    assert model_obj().Model() == 'ERA-5'


def test_dem_1(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    argDict = checkArgs(args, p)
    assert argDict['heights'][0] == 'skip'
    assert argDict['heights'][1] is None


def test_dem_2(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.heightlvs = [10, 100, 1000]
    argDict = checkArgs(args, p)
    assert argDict['heights'][0] == 'lvs'
    assert np.allclose(argDict['heights'][1], [10, 100, 1000])


def test_dem_3(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.heightlvs = [10, 100, 1000]
    args.query_area = os.path.join(SCENARIO_2, 'stations.csv')
    argDict = checkArgs(args, p)
    assert argDict['heights'][0] == 'lvs'
    assert np.allclose(argDict['heights'][1], [10, 100, 1000])


def test_dem_4(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.query_area = os.path.join(SCENARIO_2, 'stations.csv')
    argDict = checkArgs(args, p)
    assert argDict['heights'][0] == 'pandas'
    assert argDict['heights'][1][0] == argDict['wetFilenames'][0]


def test_dem_5(parsed_args):
    '''Test that passing a raster format with height levels throws an error'''
    args, p = parsed_args
    args.query_area = [os.path.join(SCENARIO_1, 'geom', 'lat.dat'), os.path.join(SCENARIO_1, 'geom', 'lat.dat')]
    argDict = checkArgs(args, p)
    assert argDict['heights'][0] == 'download'
    assert argDict['heights'][1] == os.path.join(argDict['out'], 'geom', 'warpedDEM.dem')
