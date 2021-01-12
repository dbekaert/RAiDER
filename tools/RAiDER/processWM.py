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
import matplotlib.pyplot as plt

from datetime import datetime, date

from RAiDER.logger import *
from RAiDER.utilFcns import getTimeFromFile


    
def prepareWeatherModel(
        weatherDict,
        time=None,
        wmLoc=None,
        lats=None,
        lons=None,
        ll_bounds=None,
        zref=None,
        download_only=False,
        makePlots=False
    ):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    weather_model, weather_files, weather_model_name = \
        weatherDict['type'], weatherDict['files'], weatherDict['name']

    # Ensure the file output location exists
    if wmLoc is None:
        wmLoc = os.path.join(os.getcwd(), 'weather_files')
    os.makedirs(wmLoc, exist_ok = True)

    # check whether weather model files are supplied or should be downloaded
    download_flag = True
    if weather_files is None:
        if time is None:
            raise RuntimeError(
                    'prepareWeatherModel: Either a file or a time must be specified'
                )
        weather_model.filename(time, wmLoc)
        if os.path.exists(weather_model.files[0]):
            logger.warning(
                'Weather model already exists, please remove it ("%s") if you want '
                'to download a new one.', weather_model.files
            )
            download_flag = False
    else:
        download_flag = False
        time = getTimeFromFile(weather_files[0])

    # if no weather model files supplied, check the standard location
    if download_flag:
        weather_model.fetch(*weather_model.files, lats, lons, time)

    # If only downloading, exit now
    if download_only: 
        logger.warning(
            'download_only flag selected. No further processing will happen.'
        )
        return None

    # Otherwise, load the weather model data
    weather_model.load(
            outLats=lats, 
            outLons=lons, 
            zref=zref
        )

    # Logging some basic info
    logger.debug(
        'Number of weather model nodes: {}'.format(
            np.prod(weather_model.getWetRefractivity().shape)
        )
    )
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
        plt.close('all')

    try:
        weather_model.write()
    except Exception:
        logger.exception("Unable to save weathermodel to file")
    finally:
        del weather_model

