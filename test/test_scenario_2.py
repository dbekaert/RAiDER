# Unit and other tests
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pytest
from shutil import copyfile
from test import TEST_DIR, pushd

from RAiDER.delay import tropo_delay
from RAiDER.losreader import Zenith
from RAiDER.checkArgs import modelName2Module

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_2")
_RTOL = 1e-2


@pytest.mark.long
def test_computeDelay(tmp_path):
    '''
    Scenario to use:
    2: GNSS station list
    '''
    wetName = 'stations_with_Delays.csv'
    wetFile = os.path.join(SCENARIO_DIR, wetName)
    # Not used for station file input, only passed for consistent input arguments
    hydroFile = wetFile

    true_delay = os.path.join(SCENARIO_DIR, 'ERA5_true_GNSS.csv')

    station_file = os.path.join(SCENARIO_DIR, 'stations.csv')
    copyfile(station_file, wetFile)
    stats = pd.read_csv(station_file).drop_duplicates(subset=["Lat", "Lon"])
    lats = stats['Lat'].values
    lons = stats['Lon'].values

    _, model_obj = modelName2Module('ERA5')

    with pushd(tmp_path):

        # packing the dictionairy
        args = {}
        args['los'] = Zenith
        args['lats'] = lats
        args['lons'] = lons
        args['ll_bounds'] = (33.746, 36.795, -118.312, -114.892)
        args['heights'] = ('merge', [wetFile])
        args['pnts_file'] = 'ERA5_true_GNSS_query_points.h5'
        args['flag'] = "station_file"
        args['weather_model'] = {"type": model_obj(), "files": None, "name": "ERA5"}
        args['wmLoc'] = None
        args['zref'] = 20000.
        args['outformat'] = "csv"
        args['times'] = datetime(2020, 1, 3, 23, 0, 0)
        args['out'] = tmp_path
        args['download_only'] = False
        args['wetFilenames'] = wetFile
        args['hydroFilenames'] = hydroFile
        args['verbose'] = True

        (_, _) = tropo_delay(args)

    # get the results
    est_delay = pd.read_csv(wetFile).drop_duplicates(subset=["Lat", "Lon"])
    true_delay = pd.read_csv(true_delay).drop_duplicates(subset=["Lat", "Lon"])

    # get the true delay from the weather model
    assert np.nanmax(np.abs((est_delay['wetDelay'].values - true_delay['wetDelay'].values) / true_delay['wetDelay'].values)) < _RTOL
    assert np.nanmax(np.abs((est_delay['hydroDelay'].values - true_delay['hydroDelay'].values) / true_delay['hydroDelay'].values)) < _RTOL
