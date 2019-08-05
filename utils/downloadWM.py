"""Compute the delay from a point to the transmitter.

Dry and hydrostatic delays are calculated in separate functions.
Currently we take samples every _STEP meters, which causes either
inaccuracies or inefficiencies, and we no longer can integrate to
infinity.
"""


from osgeo import gdal
gdal.UseExceptions()

# standard imports
import dask 
import itertools
import numpy as np
import os
import pyproj
import tempfile
import queue
import threading

# local imports
import constants as const
import demdownload
import losreader
import util


def downloadWMFile(weather_model_name, time, outLoc, verbose = False):
    '''
    Check whether the output weather model exists, and 
    if not, download it.
    '''
    f = os.path.join(outLoc, 'weather_files', 
        '{}_{}.nc'.format(weather_model_name, 
         dt.strftime(time, '%Y_%m_%d_T%H_%M_%S')))

    if verbose: 
       print('Storing weather model at: {}'.format(f))

    if not os.path.exists(f):
        try:
           weather_model.fetch(lats, lons, time, f)
        except Exception as e:
           print('ERROR: Unable to download weather data')
           print('Exception encountered: {}'.format(e))
           sys.exit(0)
    else:
        print('WARNING: Weather model already exists, skipping download')
    if download_only:
        print('WARNING: download_only flag selected. I will only '\
              'download the weather'\
              ' model, without doing any further processing.')
        return None, None

