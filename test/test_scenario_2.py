# Unit and other tests
from datetime import datetime
import numpy as np
import os
import unittest
import pandas as pd
from shutil import copyfile

from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay
from RAiDER.utilFcns import gdal_open, modelName2Module

class RunTests(unittest.TestCase):

    #########################################
    # Scenario to use: 
    # 1: GNSS station list
    scenario = 'scenario_2'
    weather_model_name = 'ERA5'
    los = Zenith
    download_only = False
    verbose = True
    wetName = 'stations_with_Delays.csv'

    t = datetime(2020,1,3,23,0,0)
    #########################################

    # load the weather model type and date for the given scenario
    outdir = os.path.join(os.getcwd(), 'test', scenario)
    wmLoc = os.path.join(outdir, 'weather_files')
    heights = ('DEM', os.path.join(outdir, 'geom/warpedDEM.dem'))
    wetFile = os.path.join(outdir, wetName)
    hydroFile = wetFile # Not used for station file input, only passed for consistent input arguments

    true_delay = os.path.join(outdir, 'ERA5_true_GNSS.csv')

    station_file = os.path.join(outdir, 'stations.csv')
    copyfile(station_file, wetFile)
    stats = pd.read_csv(station_file)
    lats = stats['Lat'].values
    lons = stats['Lon'].values
    ll_bounds = (33.746, 36.795, -118.313, -114.892)
    heights = ('merge', [os.path.join(outdir,wetName)])
    flag = 'station_file'
    zref = 20000.
    outformat = 'csv'

    model_module_name, model_obj = modelName2Module(weather_model_name)
    weather_model = {'type': model_obj(), 'files': None, 'name': weather_model_name}
 

    def test_computeDelay(self):
        (_,_) = tropo_delay(self.los, self.lats, self.lons, self.ll_bounds, self.heights, self.flag, 
                            self.weather_model, self.wmLoc, self.zref, self.outformat, self.t, self.outdir, 
                            self.download_only, self.verbose, self.wetFile, self.hydroFile)
 
        # get the results
        est_delay  = pd.read_csv(self.wetFile)
        true_delay = pd.read_csv(self.true_delay)

        # get the true delay from the weather model
        self.assertTrue(np.allclose(est_delay['totalDelay'].values, true_delay['totalDelay'].values, equal_nan=True))
        self.assertTrue(np.allclose(est_delay['wetDelay'].values, true_delay['wetDelay'].values, equal_nan=True))
        self.assertTrue(np.allclose(est_delay['hydroDelay'].values, true_delay['hydroDelay'].values, equal_nan=True))

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

