# Unit and other tests
import datetime
import glob
from osgeo import gdal
import numpy as np
import traceback
import os

import delay
import util
from . import data4tests as d4t

def main():
    #########################################
    # Scenario to use: 
    # 0: single point, ERAI, download DEM 
    # 1: single point, WRF, download DEM 
    # 2: 
    # 3: 
    # 4: Small area, ERAI, los available
    # 5: Small area, WRF, los available
    scenario = 'scenario_5'

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
    delay.tropo_delay(los =los, 
                     lat = latfile, 
                     lon = lonfile, 
                     heights = heights,
                     weather = wm, 
                     zref = 15000,
                     time =test_time, 
                     out = outdir,
                     parallel=False, 
                     verbose = True)

    

if __name__=='__main__':

    main()
