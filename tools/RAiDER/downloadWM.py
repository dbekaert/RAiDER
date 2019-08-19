#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# standard imports
import os
from datetime import datetime as dt

# local imports
import RAiDER.util as util


def downloadWMFile(weather_model_name, time, outLoc, verbose = False):
    '''
    Check whether the output weather model exists, and 
    if not, download it.
    '''
    util.mkdir('weather_files')
    f = os.path.join(outLoc, 
        '{}_{}.nc'.format(weather_model_name,
         dt.strftime(time, '%Y_%m_%d_T%H_%M_%S')))

    if verbose: 
       print('Storing weather model at: {}'.format(f))

    download_flag = True
    if os.path.exists(f):
       print('WARNING: Weather model already exists, skipping download')
       download_flag = False

    return download_flag, f

