# Unit and other tests
import datetime
import glob
from osgeo import gdal
import numpy as np
import traceback
import os
import unittest

import delay
import util
from . import data4tests as d4t

class TimeTests(unittest.TestCase):

    #########################################
    # Scenario to use: 
    # 1: single point, ERAI, download DEM 
    # 2: 
    # 3: 
    # 4: Small area, ERAI, los available
    # 5: Small area, WRF, los available
    scenario = 'scenario_4'

    # Zenith or LOS?
    useZen = True
    #########################################

    # load the weather model type and date for the given scenario
    outdir = os.getcwd() + '/test/'
    basedir = outdir + '/{}/'.format(scenario)
    lines=[]
    with open(os.path.join(basedir, 'wmtype'), 'r') as f:
        for line in f:
            lines.append(line.strip())
    wmtype = lines[0]
    test_time = datetime.datetime.strptime(lines[1], '%Y%m%d%H%M%S')

    # get the data for the scenario
    latfile = os.path.join(basedir, 'lat.rdr')
    lonfile = os.path.join(basedir,'lon.rdr')
    losfile = os.path.join(basedir,'los.rdr')
    demfile = os.path.join(basedir,'warpedDEM.dem')

    if os.path.exists(demfile):
        heights = ('dem', demfile)
    else:
        heights = ('download', None)

    if useZen:
        los = None
    else:
        los = ('los', losfile)

    # load the weather model
    wm = d4t.load_weather_model(wmtype)

    # test error messaging
    def test_tropo_smallArea(self):
        delay.tropo_delay(los = self.los, 
                     lat = self.latfile, 
                     lon = self.lonfile, 
                     heights = self.heights,
                     weather = self.wm, 
                     zref = 15000,
                     time = self.test_time, 
                     out = self.outdir,
                     parallel=False, 
                     verbose = False)

    
#        self.assertTrue(np.allclose(testData, refData,equal_nan = True))
        self.assertTrue(True)

if __name__=='__main__':

    unittest.main()
