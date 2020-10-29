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

def fixLL(lats, lons, weather_model):
    ''' 
    Need to correct lat/lon bounds because not all of the weather models have valid data 
    exactly bounded by -90/90 (lats) and -180/180 (lons); for GMAO and MERRA2, need to 
    adjust the longitude higher end with an extra buffer; for other models, the exact 
    bounds are close to -90/90 (lats) and -180/180 (lons) and thus can be rounded to the 
    above regions (either in the downloading-file API or subsetting-data API) without problems.
    '''
    if weather_model._Name is 'GMAO' or weather_model._Name is 'MERRA2':
        ex_buffer_lon_max = weather_model._lon_res
    else:
        ex_buffer_lon_max = 0.0

    # These are generalized for potential extra buffer in future models
    ex_buffer_lat_min = 0.0
    ex_buffer_lat_max = 0.0
    ex_buffer_lon_min = 0.0

    # The same Nextra used in the weather model base class _get_ll_bounds
    Nextra = 2
    
    # At boundary lats and lons, need to modify Nextra buffer so that the lats and lons do not exceed the boundary
    lats[lats < (-90.0 + Nextra * weather_model._lat_res + ex_buffer_lat_min)] = (-90.0 + Nextra * weather_model._lat_res + ex_buffer_lat_min)
    lats[lats > (90.0 - Nextra * weather_model._lat_res - ex_buffer_lat_max)] = (90.0 - Nextra * weather_model._lat_res - ex_buffer_lat_max)
    lons[lons < (-180.0 + Nextra * weather_model._lon_res + ex_buffer_lon_min)] = (-180.0 + Nextra * weather_model._lon_res + ex_buffer_lon_min)
    lons[lons > (180.0 - Nextra * weather_model._lon_res - ex_buffer_lon_max)] = (180.0 - Nextra * weather_model._lon_res - ex_buffer_lon_max)

    return lats, lons
