#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import contextlib
import logging
import os
import sys
from datetime import datetime

import numpy as np

from RAiDER.utilFcns import getTimeFromFile

log = logging.getLogger(__name__)


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

    log.debug('Storing weather model at: %s', f)

    download_flag = True
    if os.path.exists(f):
        log.warning('Weather model already exists, skipping download')
        download_flag = False

    return download_flag, f


def prepareWeatherModel(weatherDict, wmFileLoc, out, lats=None, lons=None,
                        los=None, zref=None, time=None,
                        download_only=False, makePlots=False):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''

    # Make weather
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
            log.exception('Unable to download weather data')
            # TODO: Is this really an appropriate place to be calling sys.exit?
            sys.exit(0)

        # exit on download if download_only requested
        if download_only:
            log.warning(
                'download_only flag selected. No further processing will happen.'
            )
            return None, None, None

    # Load the weather model data
    if weather_files is not None:
        weather_model.load(*weather_files, outLats=lats, outLons=lons, los=los, zref=zref)
        download_flag = False
    else:
        weather_model.load(f, outLats=lats, outLons=lons, los=los, zref=zref)

    log.debug('Number of weather model nodes: %d', np.prod(weather_model.getWetRefractivity().shape))
    log.debug('Shape of weather model: %s', weather_model.getWetRefractivity().shape)
    log.debug(
        'Bounds of the weather model: %.2f/%.2f/%.2f/%.2f (SNWE)',
        np.nanmin(weather_model._ys), np.nanmax(weather_model._ys),
        np.nanmin(weather_model._xs), np.nanmax(weather_model._xs)
    )
    log.debug('Weather model: %s', weather_model.Model())
    log.debug(
        'Mean value of the wet refractivity: %f',
        np.nanmean(weather_model.getWetRefractivity())
    )
    log.debug(
        'Mean value of the hydrostatic refractivity: %f',
        np.nanmean(weather_model.getHydroRefractivity())
    )
    log.debug(weather_model)

    if makePlots:
        p = weather_model.plot('wh', True)
        p = weather_model.plot('pqt', True)

    return weather_model, lats, lons
