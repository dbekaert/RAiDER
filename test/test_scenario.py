# Unit and other tests
import datetime
import gdal
import numpy as np
import os
import pandas as pd
import unittest

import RAiDER.llreader
import RAiDER.util

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
    outdir = os.getcwd() + '/test/'
    basedir = outdir + '/{}/'.format(scenario)
    lines=[]
    with open(os.path.join(basedir, 'wmtype'), 'r') as f:
        for line in f:
            lines.append(line.strip())
    wmtype = lines[0]
    test_time = datetime.datetime.strptime(lines[1], '%Y%m%d%H%M%S')
    flag = lines[-1]

    # get the data for the scenario
    if flag == 'station_file':
       filename = os.path.join(basedir, 'station_file.txt')
       [lats, lons] = llreader.readLL(filename)
    else:
       latfile = os.path.join(basedir, 'lat.rdr')
       lonfile = os.path.join(basedir,'lon.rdr')
       losfile = os.path.join(basedir,'los.rdr')
       [lats, lons] = RAiDER.llreader.readLL(latfile, lonfile)

    # DEM
    demfile = os.path.join(basedir,'warpedDEM.dem')
    wmLoc = os.path.join(basedir, 'weather_files')
    RAiDER.util.mkdir(wmLoc)
    if os.path.exists(demfile):
        heights = ('dem', demfile)
    else:
        heights = ('download', None)
    lats, lons, hts = RAiDER.llreader.getHeights(lats, lons,heights, demfile)

    if useZen:
        los = None
    else:
        los = ('los', losfile)

    # load the weather model
    model_name, wm = RAiDER.util.modelName2Module(wmtype)

    # test error messaging
    def test_tropo_smallArea(self):
        wetDelay, hydroDelay = \
            delay.tropo_delay(self.test_time, self.los, self.lats, self.lons, self.hts,
                  self.wm(), self.wmLoc, self.zref, self.out,
                  parallel=False, verbose = True,
                  download_only = False)
        totalDelayEst = wetDelay+hydroDelay
        delayDF = pd.read_csv('weather_model_data.csv')
        totalDelay = np.trapz(delayDF['totalRef'].values, x=delayDF['Z'].values)
        self.assertTrue(totalDelay==totalDelayEst)

if __name__=='__main__':

    unittest.main()

