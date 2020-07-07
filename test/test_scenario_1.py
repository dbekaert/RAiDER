# Unit and other tests
import datetime
from osgeo import gdal
import numpy as np
import os
import pickle
import unittest

from RAiDER.utilFcns import gdal_open, modelName2Module
from RAiDER.constants import Zenith
from RAiDER.delay import tropo_delay

class RunTests(unittest.TestCase):

    #########################################
    # Scenario to use: 
    # 1: Small area, ERA5, Zenith
    scenario = 'scenario_1'
    weather_model_name = 'ERA5'
    los = Zenith
    download_only = False
    verbose = True
    year = 2020
    month = 1
    day = 3
    hour = 23
    #########################################

    # load the weather model type and date for the given scenario
    outdir = os.path.join(os.getcwd(),scenario)
    out = outdir
    wmLoc = os.path.join(outdir, 'weather_files')

    true_wet = os.path.join(outdir, 'ERA5_wet_true.envi')
    true_hydro = os.path.join(outdir, 'ERA5_hydro_true.envi')

    lats = gdal_open(os.path.join(out, 'geom', 'ERA5_Lat_2018_01_01_T00_00_00.dat'))
    lons = gdal_open(os.path.join(out, 'geom', 'ERA5_Lon_2018_01_01_T00_00_00.dat'))
    ll_bounds = (15.75, 18.25, -103.24, -99.75)
    heights = ('download', os.path.join(outdir, 'geom', 'warpedDEM.dem'))
    flag = 'files'
    zref = 20000.
    outformat = 'envi'
    t = datetime.datetime(year, month, day, hour, 0)

    model_module_name, model_obj = modelName2Module(weather_model_name)
    weather_model = {'type': model_obj(), 'files': None, 'name': weather_model_name}
 
    wfn = '{}_wet_{}-{:02}-{:02}T{}:00:00_std.{}'.format(weather_model_name,year, month, day, hour,outformat)
    hfn = '{}_hydro_{}-{:02}-{:02}T{}:00:00_std.{}'.format(weather_model_name,year,month,day, hour,outformat)
    wetFile = os.path.join(out, wfn)
    hydroFile = os.path.join(out, hfn)

    def test_computeDelay(self):
        (_,_) = tropo_delay(self.los, self.lats, self.lons, self.ll_bounds, self.heights, self.flag, 
                            self.weather_model, self.wmLoc, self.zref, self.outformat, self.t, self.out, self.download_only, 
                            self.verbose, self.wetFile, self.hydroFile)
 
        # get the results
        wet = gdal_open(self.wetFile)
        hydro = gdal_open(self.hydroFile)
        true_wet = gdal_open(self.true_wet)
        true_hydro = gdal_open(self.true_hydro)

        # get the true delay from the weather model
        self.assertTrue(np.allclose(wet, true_wet, equal_nan=True))
        self.assertTrue(np.allclose(hydro, true_hydro, equal_nan=True))

def main():
    unittest.main()
   
if __name__=='__main__':

    unittest.main()

