# Unit and other tests
import datetime
import gdal
import numpy as np
import os
import pandas as pd
import pickle
import unittest

from RAiDER.delay import computeDelay, interpolateDelay
from RAiDER.llreader import readLL, getHeights
from RAiDER.losreader import getLookVectors
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import pickle_load, modelName2Module

@unittest.skipIf('skipping full model test until all other unit tests pass')
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
    wmLoc = os.path.join(basedir, 'weather_files')

    data = pickle_load(os.path.join(basedir, 'data.pik'))
    lats,lons,los,zref,hgts = data['lats'], data['lons'], data['los'], data['zref'], data['hgts']
    if zref < 30000:
       zref = 30000

    #weather_model = pickle_load(os.path.join(basedir, 'pickledweathermodel.pik'))
    model_module_name, model_obj = modelName2Module('ERA5')
    era5 = {'type': model_obj(), 'files': None, 'name': 'ERA5'}
    weather_model, lats, lons= prepareWeatherModel(era5,wmLoc, basedir, lats = lats, lons = lons, time = datetime.datetime(2019,1,1,2,0,0), verbose=True)

    # Compute the true delay
    wrf = weather_model._wet_refractivity[1,1,:]
    hrf = weather_model._hydrostatic_refractivity[1,1,:]
    zs = weather_model._zs
    mask = (zs > 2907) & (zs < zref) & (~np.isnan(wrf) & ~np.isnan(hrf))
    zt = np.linspace(2907, zref, 1000)
    wt = np.interp(zt, zs[mask], wrf[mask])
    ht = np.interp(zt, zs[mask], hrf[mask])
    wetDelay = 1e-6*np.trapz(wt,zt) 
    hydroDelay = 1e-6*np.trapz(ht, zt)
    delays = interpolateDelay(weather_model, lats, lons, hgts,los, zref = zref, useWeatherNodes = False)

    # test error messaging
    @unittest.skip("skipping full model test until all other unit tests pass")
    def test_computeDelay(self):
        wetDelay, hydroDelay = \
            computeDelay(self.los, self.lats, self.lons, self.hgts,
                  self.weather_model, self.zref, self.out,
                  parallel=False, verbose = True)

        # get the true delay from the weather model
        self.assertTrue(np.abs(self.delays[0]-wetDelay) < 0.01)
        self.assertTrue(np.abs(self.delays[1]-hydroDelay)< 0.01)
        self.assertTrue(np.abs(self.wetDelay - wetDelay)/wetDelay < 0.1)
        self.assertTrue(np.abs(self.hydroDelay - hydroDelay)/hydroDelay < 0.3)

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

