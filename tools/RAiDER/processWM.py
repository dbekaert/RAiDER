#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import contextlib
import os
import sys

import numpy as np

<<<<<<< HEAD
from RAiDER.logger import *
from RAiDER.utilFcns import getTimeFromFile

=======
from datetime import datetime

from RAiDER.logger import *
from RAiDER.utilFcns import getTimeFromFile

>>>>>>> Update unit tests, fix bugs, and clean up

def getWMFilename(weather_model_name, time, outLoc):
    '''
    Check whether the output weather model exists, and
    if not, download it.
    '''
    with contextlib.suppress(FileExistsError):
        os.mkdir('weather_files')

    f = os.path.join(
        outLoc,
        '{}_{}.nc'.format(
            weather_model_name,
            datetime.strftime(time, '%Y_%m_%d_T%H_%M_%S')
        )
    )

    logger.debug('Storing weather model at: %s', f)

    if os.path.exists(f):
        logger.warning('Weather model already exists, skipping download')
<<<<<<< HEAD
        download_flag = False
=======
>>>>>>> Update unit tests, fix bugs, and clean up

    return f


def prepareWeatherModel(weather_model, wmFileLoc, lats=None, lons=None,
                        los=None, zref=None, time=None,
                        download_only=False, makePlots=False):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''

    # Make weather
<<<<<<< HEAD
    weather_model, weather_files, weather_model_name = \
        weatherDict['type'], weatherDict['files'], weatherDict['name']

    # check whether weather model files are supplied
    if weather_files is None:
        download_flag, f = getWMFilename(weather_model.Model(), time, wmFileLoc)
    else:
        download_flag = False
        time = getTimeFromFile(weather_files[0])

    # if no weather model files supplied, check the standard location
    if download_flag:
        try:
            weather_model.fetch(lats, lons, time, f)
        except Exception:
            logger.exception('Unable to download weather data')
            # TODO: Is this really an appropriate place to be calling sys.exit?
            sys.exit(0)

        # exit on download if download_only requested
        if download_only:
            logger.warning(
                'download_only flag selected. No further processing will happen.'
            )
            return None, None, None
=======
    if weather_model.files is not None:
        time = getTimeFromFile(weather_model.files[0])

    # Download the weather model file unless it already exists
    f = getWMFilename(weather_model.Model(), time, wmFileLoc)
    if ~os.path.exists(f):
        weather_model.fetch(lats, lons, time, f)

    # exit on download if download_only requested
    if download_only:
        logger.warning(
            'download_only flag selected. No further processing will happen.'
        )
        return None, None, None
>>>>>>> Update unit tests, fix bugs, and clean up

    # Load the weather model data
    if weather_model.files is not None:
        weather_model.load(*weather_model.files, outLats=lats, outLons=lons, los=los, zref=zref)
        download_flag = False
    else:
        weather_model.load(f, outLats=lats, outLons=lons, los=los, zref=zref)

    logger.debug('Number of weather model nodes: %d', np.prod(weather_model.getWetRefractivity().shape))
    logger.debug('Shape of weather model: %s', weather_model.getWetRefractivity().shape)
    logger.debug(
        'Bounds of the weather model: %.2f/%.2f/%.2f/%.2f (SNWE)',
        np.nanmin(weather_model._ys), np.nanmax(weather_model._ys),
        np.nanmin(weather_model._xs), np.nanmax(weather_model._xs)
    )
    logger.debug('Weather model: %s', weather_model.Model())
    logger.debug(
        'Mean value of the wet refractivity: %f',
        np.nanmean(weather_model.getWetRefractivity())
    )
    logger.debug(
        'Mean value of the hydrostatic refractivity: %f',
        np.nanmean(weather_model.getHydroRefractivity())
    )
    logger.debug(weather_model)

    if makePlots:
        p = weather_model.plot('wh', True)
        p = weather_model.plot('pqt', True)

    return weather_model, lats, lons
