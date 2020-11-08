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

from datetime import datetime

from RAiDER.logger import *
from RAiDER.utilFcns import getTimeFromFile


def getWMFilename(weather_model_name, time, outLoc):
    '''
    Check whether the output weather model exists, and
    if not, download it.
    '''
    with contextlib.suppress(FileExistsError):
        os.mkdir('weather_files')

    download_flag = True
    f = os.path.join(
        outLoc,
        '{}_{}.nc'.format(
            weather_model_name,
            datetime.strftime(time, '%Y_%m_%d_T%H_%M_%S')
        )
    )

    if weather_model_name == 'GMAO' or weather_model_name == 'MERRA2':
        f = f[:-2]+'h5'

    logger.debug('Storing weather model at: %s', f)

    if os.path.exists(f):
        logger.warning('Weather model already exists, skipping download')
        download_flag = False

    return download_flag, f


def prepareWeatherModel(
        weatherDict,
        wmFileLoc,
        lats=None,
        lons=None,
        los=None,
        zref=None,
        time=None,
        download_only=False,
        makePlots=False
    ):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    weather_model, weather_files, weather_model_name = \
        weatherDict['type'], weatherDict['files'], weatherDict['name']

    # check whether weather model files are supplied
    if weather_files is None:
        if time is None:
            raise RuntimeError('prepareWeatherModel: Either a file or a time must be specified')
        download_flag,f = getWMFilename(weather_model.Model(), time, wmFileLoc)
        weather_model.files = [f]
    else:
        download_flag = False
        time = getTimeFromFile(weather_files[0])
    
    if (time < datetime(2013, 6, 26, 0, 0, 0)) and (weather_model._Name is 'HRES'):
        weather_model.update_a_b()
    
    # if no weather model files supplied, check the standard location
    if download_flag:
        weather_model.fetch(*weather_model.files, lats, lons, time)

        # exit on download if download_only requested
        if download_only:
            logger.warning(
                'download_only flag selected. No further processing will happen.'
            )
            return None, None, None

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
