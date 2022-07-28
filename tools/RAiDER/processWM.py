#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from typing import List, Optional
import datetime as dt
from pathlib import Path
from RAiDER.types import Lats, Lons, WeatherDict

import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from RAiDER.logger import logger
from RAiDER.utilFcns import getTimeFromFile
from RAiDER.models.weatherModel import make_raw_weather_data_filename


def prepareWeatherModel(
    weatherDict: WeatherDict,
    times: List[dt.datetime]=None,
    wmLoc: Path=None,
    lats: Lats=None,
    lons: Lons=None,
    zref: float=None,
    download_only: bool=False,
    makePlots: bool=False,
    force_download: bool=False,
) -> Optional[List[Path]]:
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
    weather_model, weather_files = (weatherDict['type'], weatherDict['files'])
    weather_model.files = weather_files

    # Ensure the file output location exists
    if wmLoc is None:
        wmLoc = os.path.join(os.getcwd(), 'weather_files')
    os.makedirs(wmLoc, exist_ok=True)

    # check whether weather model files are supplied or should be downloaded
    download_flag = True
    if weather_model.files is None:
        if times is None:
            raise RuntimeError(
                'prepareWeatherModel: Either a file or a time must be specified'
            )
        weather_model.filename(times, wmLoc)
        if os.path.exists(weather_model.files[0]):
            if force_download:
                logger.warning(
                    '(Force download) Weather model already exists: "%s" but '
                    'will be re-downloaded.',
                    weather_model.files
                )
            else:
                logger.warning(
                    'Weather model already exists: "%s". Skipping download.',
                    weather_model.files
                )
                download_flag = False
    else:
        download_flag = False

    # if no weather model files supplied, check the standard location
    if download_flag:
        weather_model.fetch(weather_model.files[0], lats, lons, times)
    else:
        time = getTimeFromFile(weather_model.files[0])
        weather_model.setTime(time)
        containment = weather_model.checkContainment(lats, lons)

        if not containment:
            logger.error(
                'The weather model passed does not cover all of the input '
                'points; you need to download a larger area.'
            )
            raise RuntimeError(
                'The weather model passed does not cover all of the input '
                'points; you need to download a larger area.'
            )

    # Split the data from the range into one file per day and update the
    # weather model with the new files
    if len(times) > 1:
        date_range_filename = weather_model.files[0]
        weather_model.files = []
        for i, time in enumerate(times):
            with xr.open_dataset(date_range_filename) as block:
                block["z"] = block.z[i]
                block["q"] = block.q[i]
                block["t"] = block.t[i]
                single_date_filename = make_raw_weather_data_filename(
                    wmLoc,
                    weather_model._Name,
                    [time],
                )
                block.to_netcdf(single_date_filename)
                weather_model.files.append(single_date_filename)
        os.remove(date_range_filename)

    # If only downloading, exit now
    if download_only:
        logger.warning(
            'download_only flag selected. No further processing will happen.'
        )
        return None

    # Otherwise, process the weather model data
    out_files: List[Path] = []
    for filename, time in zip(weather_model.files, times):
        weather_model.setTime(time)
        processed_data_filename = weather_model.load(
            filename,
            wmLoc,
            outLats=lats,
            outLons=lons,
            zref=zref,
        )
        if processed_data_filename is not None:
            logger.warning(
                'Skipping processing for %s because it is already processed',
                filename
            )
            out_files.append(processed_data_filename)
            continue

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
            processed_data_filename = weather_model.write()
            out_files.append(processed_data_filename)
        except Exception as e:
            logger.exception("Unable to save weathermodel to file")
            logger.exception(e)
            raise RuntimeError("Unable to save weathermodel to file")
    return out_files


# def checkBounds(weather_model, outLats, outLons):
#     '''Check the bounds of a weather model'''
#     ds = xr.load_dataset(weather_model.files[0])  # TODO: xr is undefined
#     coords = ds.coords  # coords is dict-like
#     keys = [k for k in coords.keys()]
#     xc = coords[keys[0]]
#     yc = coords[keys[1]]
#     lat_bounds = [yc.min(), yc.max()]
#     lon_bounds = [xc.min(), xc.max()]
#     self_extent = lat_bounds + lon_bounds
#     in_extent = weather_model._getExtent(outLats, outLons)

#     return in_extent, self_extent


def weather_model_debug(
    los,
    lats,
    lons,
    ll_bounds,
    weather_model,
    wmLoc,
    zref,
    times,
    out,
    download_only
):
    """
    raiderWeatherModelDebug main function.
    """

    logger.debug('Starting to run the weather model calculation with debugging plots')
    logger.debug('Times type: %s', type(times))
    logger.debug('Times: %s', times[0].strftime('%Y%m%d') if len(times) == 1 else times[0].strftime('%Y%m%d') + '/' + times[-1].strftime('%Y%m%d'))

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    # TODO: make_weather_model_filename is undefined
    wm_filename = make_weather_model_filename(
        weather_model['name'],
        times,
        ll_bounds
    )
    weather_model_file = os.path.join(wmLoc, wm_filename)

    if not os.path.exists(weather_model_file):
        prepareWeatherModel(
            weather_model,
            times,
            wmLoc=wmLoc,
            lats=lats,
            lons=lons,
            ll_bounds=ll_bounds,
            zref=zref,
            download_only=download_only,
            makePlots=True
        )
        try:
            weather_model.write2NETCDF4(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")
