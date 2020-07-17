# Unit and other tests
from datetime import datetime
import numpy as np
import os
import pandas as pd
import pytest
from shutil import copyfile
from test import DATA_DIR, TEST_DIR, pushd

from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open, modelName2Module

SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_2")

def test_computeDelay(tmp_path):
    '''
    Scenario to use: 
    2: GNSS station list
    '''
    wetName = 'stations_with_Delays.csv'
    wetFile = os.path.join(SCENARIO_DIR, wetName)
    hydroFile = wetFile # Not used for station file input, only passed for consistent input arguments

    # load the weather model type and date for the given scenario
    wmLoc = os.path.join(SCENARIO_DIR, 'weather_files')

    true_delay = os.path.join(SCENARIO_DIR, 'ERA5_true_GNSS.csv')

    station_file = os.path.join(SCENARIO_DIR, 'stations.csv')
    copyfile(station_file, wetFile)
    stats = pd.read_csv(station_file)
    lats = stats['Lat'].values
    lons = stats['Lon'].values
    
    _, model_obj = modelName2Module('ERA5')
 

    with pushd(tmp_path):
        (_,_) = tropo_delay(
           los=Zenith, 
           lats=lats, 
           lons=lons, 
           ll_bounds=(33.746, 36.795, -118.313, -114.892),
           heights=('merge', [wetFile]), 
           flag='station_file', 
           weather_model={'type': model_obj(), 'files': None, 'name': 'ERA5'},
           wmLoc=None, 
           zref=20000., 
           outformat='csv', 
           time=datetime(2020,1,3,23,0,0), 
           out=tmp_path, 
           download_only=False, 
           verbose=True, 
           wetFilename=wetFile, 
           hydroFilename=hydroFile
           )
 
    # get the results
    est_delay  = pd.read_csv(wetFile)
    true_delay = pd.read_csv(true_delay)

    # get the true delay from the weather model
    assert np.allclose(est_delay['totalDelay'].values, true_delay['totalDelay'].values, equal_nan=True)
    assert np.allclose(est_delay['wetDelay'].values, true_delay['wetDelay'].values, equal_nan=True)
    assert np.allclose(est_delay['hydroDelay'].values, true_delay['hydroDelay'].values, equal_nan=True)

