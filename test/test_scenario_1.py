import datetime
import os
import pytest
import urllib.error

import numpy as np

from test import TEST_DIR, pushd

from pathlib import Path
from test import DATA_DIR, TEST_DIR, pushd

from RAiDER.losreader import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open
from RAiDER.checkArgs import makeDelayFileNames, modelName2Module

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")
_RTOL = 1e-2


@pytest.mark.long
def test_tropo_delay_ERAI(tmp_path):
    '''
    Scenario:
    1: Small area, ERAI, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERAI")


@pytest.mark.long
def test_tropo_delay_ERA5(tmp_path):
    '''
    Scenario:
    1: Small area, ERA5, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERA5")


@pytest.mark.long
def test_tropo_delay_ERA5T(tmp_path):
    '''
    Scenario:
    1: Small area, ERA5T, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERA5T")


@pytest.mark.long
def test_tropo_delay_GMAO(tmp_path):
    '''
    Scenario:
    1: Small area, GMAO, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="GMAO")


@pytest.mark.long
def test_tropo_delay_MERRA2(tmp_path):
    '''
    Scenario:
    1: Small area, MERRA2, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="MERRA2")


@pytest.mark.skip(reason="NCMR keeps hanging")
def test_tropo_delay_NCMR(tmp_path):
    '''
    Scenario:
    1: Small area, NCMR, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="NCMR")


@pytest.mark.long
def test_tropo_delay_GMAO(tmp_path):
    '''
    Scenario:
    1: Small area, GMAO, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="GMAO")


def core_test_tropo_delay(tmp_path, modelName):
    '''
    Scenario: 
    1: Small area, Zenith delay
    '''
    lats = gdal_open(os.path.join(
        SCENARIO_DIR, 'geom', 'lat.dat'
    ))
    lons = gdal_open(os.path.join(
        SCENARIO_DIR, 'geom', 'lon.dat'
    ))

    if modelName == 'ERAI':
        time = datetime.datetime(2018, 1, 3, 23, 0)
    elif modelName == 'NCMR':
        time = datetime.datetime(2018, 7, 1, 0, 0)
    else:
        time = datetime.datetime(2020, 1, 3, 23, 0)

    wmLoc = os.path.join(SCENARIO_DIR, 'weather_files')
    if not os.path.exists(wmLoc):
        os.mkdir(wmLoc)

    _, model_obj = modelName2Module(modelName)
    wet_file, hydro_file = makeDelayFileNames(
        time, Zenith, "envi", modelName, tmp_path
    )

    with pushd(tmp_path):
        # packing the dictionairy
        args = {}
        args['los'] = Zenith
        args['lats'] = lats
        args['lons'] = lons
        args['ll_bounds'] = (15.75, 18.25, -103.24, -99.75)
        args['heights'] = ("dem", os.path.join(TEST_DIR, "test_geom", "warpedDEM.dem"))
        args['pnts_file'] = 'lat_query_points.h5'
        args['flag'] = "files"
        args['weather_model'] = {"type": model_obj(), "files": None, "name": modelName}
        args['wmLoc'] = wmLoc
        args['zref'] = 20000.
        args['outformat'] = "envi"
        args['times'] = time
        args['out'] = tmp_path
        args['download_only'] = False
        args['wetFilenames'] = wet_file
        args['hydroFilenames'] = hydro_file
        args['verbose'] = True

        (_, _) = tropo_delay(args)

        # get the results
        wet = gdal_open(wet_file)
        hydro = gdal_open(hydro_file)
        true_wet = gdal_open(
            os.path.join(
                SCENARIO_DIR,
                modelName + "/wet.envi"
            ),
            userNDV=0.
        )
        true_hydro = gdal_open(
            os.path.join(
                SCENARIO_DIR,
                modelName + "/hydro.envi"
            ),
            userNDV=0.
        )

        # get the true delay from the weather model
        assert np.nanmax(np.abs((wet - true_wet) / true_wet)) < _RTOL
        assert np.nanmax(np.abs((hydro - true_hydro) / true_hydro)) < _RTOL
