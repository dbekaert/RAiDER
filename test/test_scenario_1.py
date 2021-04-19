import datetime
import os
from test import DATA_DIR, TEST_DIR, pushd
import urllib.error

import numpy as np
import pytest


from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open, makeDelayFileNames, modelName2Module

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")
_RTOL = 1e-4


@pytest.mark.timeout(300)
def test_tropo_delay_ERA5(tmp_path):
    '''
    Scenario:
    1: Small area, ERA5, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERA5")


@pytest.mark.timeout(300)
def test_tropo_delay_GMAO(tmp_path):
    '''
    Scenario:
    1: Small area, GMAO, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="GMAO")

# comment out MERRA-2 test for now: it passes on local machines but not in CircleCI. Need further look into this.
# def test_tropo_delay_MERRA2(tmp_path):
#    '''
#    Scenario:
#    1: Small area, MERRA2, Zenith delay
#    '''
#    core_test_tropo_delay(tmp_path, modelName="MERRA2")


@pytest.mark.timeout(300)
def test_tropo_delay_HRES(tmp_path):
    '''
    Scenario:
    1: Small area, HRES, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="HRES")


@pytest.mark.timeout(300)
def test_tropo_delay_ERA5T(tmp_path):
    '''
    Scenario:
    1: Small area, ERA5T, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERA5T")


@pytest.mark.timeout(300)
def test_tropo_delay_ERAI(tmp_path):
    '''
    Scenario:
    1: Small area, ERAI, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="ERAI")

@pytest.mark.xfail(
        raises=urllib.error.URLError
    )
def test_tropo_delay_NCMR(tmp_path):
    '''
    Scenario:
    1: Small area, NCMR, Zenith delay
    '''
    core_test_tropo_delay(tmp_path, modelName="NCMR")


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
        assert np.allclose(
            wet,
            true_wet,
            equal_nan=True,
            rtol=_RTOL
        )
        assert np.allclose(
            hydro,
            true_hydro,
            equal_nan=True,
            rtol=_RTOL
        )
