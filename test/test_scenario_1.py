import datetime
import os
from test import DATA_DIR, TEST_DIR, pushd

import numpy as np
import pytest

from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open, makeDelayFileNames, modelName2Module

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")
_RTOL = 1e-4


def test_tropo_delay(tmp_path):
    '''
    Scenario: 
    1: Small area, ERA5, Zenith delay
    '''
    lats = gdal_open(os.path.join(
        SCENARIO_DIR, 'geom', 'lat.dat'
    ))
    lons = gdal_open(os.path.join(
        SCENARIO_DIR, 'geom', 'lon.dat'
    ))

    time = datetime.datetime(2020, 1, 3, 23, 0)

    wmLoc = os.path.join(SCENARIO_DIR, 'weather_files')
    if not os.path.exists(wmLoc):
        os.mkdir(wmLoc)

    _, model_obj = modelName2Module("ERA5")
    wet_file, hydro_file = makeDelayFileNames(
        time, Zenith, "envi", "ERA5", tmp_path
    )

    with pushd(tmp_path):
        # packing the dictionairy
        args={}
        args['los']=Zenith
        args['lats']=lats
        args['lons']=lons
        args['ll_bounds']=(15.75, 18.25, -103.24, -99.75)
        args['heights']=("dem", os.path.join(TEST_DIR, "test_geom", "warpedDEM.dem"))
        args['flag']="files"
        args['weather_model']={"type": model_obj(),"files": None,"name": "ERA5"}
        args['wmLoc']=wmLoc
        args['zref']=20000.
        args['outformat']="envi"
        args['times']=time
        args['out']=tmp_path
        args['download_only']=False
        args['wetFilenames']=wet_file
        args['hydroFilenames']=hydro_file
        
        (_, _) = tropo_delay(args)
      
        # get the results
        wet = gdal_open(wet_file)
        hydro = gdal_open(hydro_file)
        true_wet = gdal_open(
            os.path.join(
                SCENARIO_DIR,
                "wet.envi"
            ),
            userNDV=0.
        )
        true_hydro = gdal_open(
            os.path.join(
                SCENARIO_DIR,
                "hydro.envi"
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
