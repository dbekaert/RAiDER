# standard imports
import os
from datetime import datetime as dt

# local imports
import utils.util as util


def downloadWMFile(weather_model_name, time, outLoc, verbose = False):
    '''
    Check whether the output weather model exists, and 
    if not, download it.
    '''
    util.mkdir('weather_files')
    f = os.path.join(outLoc, 'weather_files', 
        '{}_{}.nc'.format(weather_model_name,
         dt.strftime(time, '%Y_%m_%d_T%H_%M_%S')))

    if verbose: 
       print('Storing weather model at: {}'.format(f))

    download_flag = True
    if os.path.exists(f):
       print('WARNING: Weather model already exists, skipping download')
       download_flag = False

    return download_flag, f

