import datetime
from test import DATA_DIR, TEST_DIR, pushd

import numpy as np

from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open, makeDelayFileNames, modelName2Module
from RAiDER.rays import ZenithLVGenerator

SCENARIO_DIR = TEST_DIR / "scenario_1"


def test_tropo_delay(tmp_path):
    '''
    Scenario:
    1: Small area, ERA5, Zenith delay
    '''
    lats = gdal_open(SCENARIO_DIR / 'geom' / 'lat.dat')
    lons = gdal_open(SCENARIO_DIR / 'geom' / 'lon.dat')

    time = datetime.datetime(2020, 1, 3, 23, 0)

    wmLoc = SCENARIO_DIR / 'weather_files'
    wmLoc.mkdir(exist_ok=True)

    zref = 20000.

    _, model_obj = modelName2Module("ERA5")
    wet_file, hydro_file = makeDelayFileNames(
        time, Zenith, "envi", "ERA5", tmp_path
    )

    with pushd(tmp_path):
        (_, _) = tropo_delay(
            losGen=ZenithLVGenerator(),
            lats=lats,
            lons=lons,
            ll_bounds=(15.75, 18.25, -103.24, -99.75),
            heights=("download", DATA_DIR / "geom" / "warpedDEM.dem"),
            flag="files",
            weather_model={
                "type": model_obj(),
                "files": None,
                "name": "ERA5"
            },
            wmLoc=wmLoc,
            zref=zref,
            outformat="envi",
            time=time,
            out=tmp_path,
            download_only=False,
            wetFilename=wet_file,
            hydroFilename=hydro_file
        )

        # get the results
        wet = gdal_open(wet_file)
        hydro = gdal_open(hydro_file)
        true_wet = gdal_open(SCENARIO_DIR / "wet.envi", userNDV=0.)
        true_hydro = gdal_open(SCENARIO_DIR / "hydro.envi", userNDV=0.)

        # get the true delay from the weather model
        assert np.allclose(wet, true_wet, equal_nan=True)
        assert np.allclose(hydro, true_hydro, equal_nan=True)
