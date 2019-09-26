# Unit and other tests
import datetime
import gdal
import numpy as np
import os
import pandas as pd
import pickle
import unittest

import RAiDER.delay
from RAiDER.llreader import readLL, getHeights
from RAiDER.losreader import getLookVectors
from RAiDER.utilFcns import pickle_load

class TimeTests(unittest.TestCase):

    #########################################
    # Scenario to use: 
    # 0: single point, fixed data
    # 1: single point, WRF, download DEM 
    # 2: 
    # 3: 
    # 4: Small area, ERAI
    # 5: Small area, WRF, los available
    # 6: Small area, ERA5, early date, Zenith
    # 7: Small area, ERA5, late date, Zenith
    # 8: Small area, ERAI, late date, Zenith
    scenario = 'scenario_0'

    # Zenith or LOS?
    useZen = True
    #########################################

    # load the weather model type and date for the given scenario
    outdir = os.path.join(os.getcwd(),'test')
    basedir = os.path.join(outdir, '{}'.format(scenario))
    out = os.path.join(basedir, os.sep)

    data = pickle_load(os.path.join(basedir, 'data.pik'))
    lats,lons,los,zref,hgts = data['lats'], data['lons'], data['los'], data['zref'], data['hgts']

    weather_model = pickle_load(os.path.join(basedir, 'pickledWeatherModel.pik'))

    # Compute the true delay
    wrf = weather_model._wet_refractivity[1,1,:]
    hrf = weather_model._hydrostatic_refractivity[1,1,:]
    zs = weather_model._zs[1,1,:]
    mask = zs > 2907
    totalDelay = 1e-6*(np.trapz(wrf[mask], zs[mask]) + np.trapz(hrf[mask], zs[mask])) 

    # test error messaging
    #@unittest.skip("skipping full model test until all other unit tests pass")
    def test_tropoSmallArea(self):
        wetDelay, hydroDelay = \
            RAiDER.delay.computeDelay(self.los, self.lats, self.lons, self.hgts,
                  self.weather_model, self.zref, self.out,
                  parallel=False, verbose = True)
        totalDelayEst = wetDelay+hydroDelay

        # get the true delay from the weather model
        self.assertTrue(np.abs(self.totalDelay - totalDelayEst) < 0.001)

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

