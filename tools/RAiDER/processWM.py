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
import xarray as xr

from datetime import datetime, date

from RAiDER.logger import *
from RAiDER.utilFcns import getTimeFromFile


def prepareWeatherModel(
    weatherDict,
    time=None,
    wmLoc=None,
    lats=None,
    lons=None,
    zref=None,
    download_only=False,
    makePlots=False
):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    weather_model, weather_files, weather_model_name = \
        weatherDict['type'], weatherDict['files'], weatherDict['name']
    weather_model.files = weather_files

    # Ensure the file output location exists
    if wmLoc is None:
        wmLoc = os.path.join(os.getcwd(), 'weather_files')
    os.makedirs(wmLoc, exist_ok=True)

    # check whether weather model files are supplied or should be downloaded
    download_flag = True
    if weather_model.files is None:
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

    # if no weather model files supplied, check the standard location
    if download_flag:
        weather_model.fetch(*weather_model.files, lats, lons, time)
    else:
        time = getTimeFromFile(weather_model.files[0])
        weather_model.setTime(time)
        checkBounds(weather_model, lats, lons)

    # If only downloading, exit now
    if download_only:
        logger.warning(
            'download_only flag selected. No further processing will happen.'
        )
        return None

    # Otherwise, load the weather model data
    f = weather_model.load(
        wmLoc,
        outLats=lats,
        outLons=lons,
        zref=zref,
    )
    if f is not None:
        logger.warning(
            'The processed weather model file already exists,'
            ' so I will use that.'
        )
        return f

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
        f = weather_model.write()
        return f
    except Exception as e:
        logger.exception("Unable to save weathermodel to file")
        logger.exception(e)
        raise RuntimeError("Unable to save weathermodel to file")
    finally:
        del weather_model


def checkBounds(weather_model, outLats, outLons):
    '''Check the bounds of a weather model'''

    ds = xr.load_dataset(weather_model.files[0])
    coords = ds.coords  # coords is dict-like
    keys = [k for k in coords.keys()]
    xc = coords[keys[0]]
    yc = coords[keys[1]]
    lat_bounds = [yc.min(), yc.max()]
    lon_bounds = [xc.min(), xc.max()]
    self_extent = lat_bounds + lon_bounds
    in_extent = weather_model._getExtent(outLats, outLons)

    if weather_model._isOutside(in_extent, self_extent):
        logger.info('Extent of the input lats/lons is: {}'.format(in_extent))
        logger.info('Extent of the weather model is: {}'.format(self_extent))
        logger.error(
            'The weather model passed does not cover all of the input '
            'points; you need to download a larger area.'
        )
        raise RuntimeError(
            'The weather model passed does not cover all of the input '
            'points; you need to download a larger area.'
        )
