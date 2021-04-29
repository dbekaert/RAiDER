import os
import pytest
import sys

import multiprocessing as mp

from argparse import ArgumentParser
from test import pushd

import RAiDER.runProgram

from RAiDER.checkArgs import checkArgs
from RAiDER.constants import Zenith, _ZREF


def isWriteable(dirpath):
    '''Test whether a directory is writeable'''
    try:
        filehandle = open(os.path.join(dirpath, 'tmp.txt'), 'w' )
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
        #'--latlon', 'latfile.dat', 'lonfile.dat',
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
    argDict = checkArgs(args, p)
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
    assert argDict['flag'] == 'bounding_box'# default arguments use a bounding box

