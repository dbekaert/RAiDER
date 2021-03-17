#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
import rasterio
from RAiDER.logger import logger
from RAiDER.utilFcns import getTimeFromFile
from RAiDER.models import weatherModel
import matplotlib.pyplot as plt
from shapely.geometry import box


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
    weather_model, weather_files = (weatherDict['type'],
                                    weatherDict['files']
                                    )
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
                'Weather model already exists, '
                'please remove it ("%s") if you want '
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
        containment = checkContainment(weather_model, lats, lons)

        if not containment:
            logger.error(
                'The weather model passed does not cover all of the input '
                'points; you need to download a larger area.'
            )
            raise RuntimeError(
                'The weather model passed does not cover all of the input '
                'points; you need to download a larger area.'
            )

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
    shape = weather_model.getWetRefractivity().shape
    logger.debug(f'Shape of weather model: {shape}')
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
        weather_model.plot('wh', True)
        weather_model.plot('pqt', True)
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


def checkContainment(weather_model: weatherModel,
                     outLats: np.ndarray,
                     outLons: np.ndarray) -> bool:
    '''Check if weather model contains bounding box of outLats and outLons'''
    weather_model_path = weather_model.files[0]
    with rasterio.open(f'netcdf:{weather_model_path}') as ds:
        datasets = ds.subdatasets

    with rasterio.open(datasets[0]) as ds:
        bounds = ds.bounds

    xmin, ymin, xmax, ymax = tuple(bounds)
    weather_model_box = box(xmin, ymin, xmax, ymax)

    xmin_input, xmax_input = np.min(outLons), np.max(outLons)
    ymin_input, ymax_input = np.min(outLats), np.max(outLats)
    input_box = box(xmin_input, ymin_input, xmax_input, ymax_input)

    # Logger
    input_box_str = [f'{x:1.2f}' for x in [xmin_input, ymin_input,
                                           xmax_input, ymax_input]]
    weath_box_str = [f'{x:1.2f}' for x in [xmin, ymin, xmax, ymax]]

    weath_box_str = ', '.join(weath_box_str)
    input_box_str = ', '.join(input_box_str)

    logger.info(f'Extent of the weather model lats/lons is: {weath_box_str}')
    logger.info(f'Extent of the input lats/lons is: {input_box_str}')

    return weather_model_box.contains(input_box)
