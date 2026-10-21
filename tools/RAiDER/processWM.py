# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The module name is camelCase, but renaming it would break every import of it.
# ruff: noqa: N999
"""Download and prepare weather model data for delay calculations."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from RAiDER.logger import logger
from RAiDER.models.customExceptions import (
    CriticalError,
    DatetimeOutsideRange,
    ExistingWeatherModelTooSmall,
    TryToKeepGoingError,
)
from RAiDER.models.weatherModel import WeatherModel, checkContainment_raw, make_raw_weather_data_filename


def prepareWeatherModel(
    weather_model: WeatherModel,
    time: dt.datetime,
    ll_bounds: list | tuple | np.ndarray,
    download_only: bool = False,
    makePlots: bool = False,
    force_download: bool = False,
) -> str | None:
    """Parse inputs to download and prepare a weather model grid for interpolation.

    Args:
        weather_model: WeatherModel   - instantiated weather model object
        time: datetime                - Python datetime to request. Will be rounded to nearest available time
        ll_bounds: list/array         - SNWE bounds target area to ensure weather model contains them
        download_only: bool           - False if preprocessing weather model data
        makePlots: bool               - whether to write debug plots
        force_download: bool          - True if you want to download even when the weather model exists

    Returns:
        str: filename of the netcdf file to which the weather model has been written
    """
    # set the bounding box from the in the case that it hasn't been set
    if weather_model.get_latlon_bounds() is None:
        weather_model.set_latlon_bounds(ll_bounds)

    # Ensure the file output location exists
    wmLoc = weather_model.get_wmLoc()
    weather_model.setTime(time)

    # get the path to the less processed weather model file
    path_wm_raw = make_raw_weather_data_filename(wmLoc, weather_model.Model(), time)

    # get the path to the more processed (cropped) weather model file
    path_wm_crop = weather_model.out_file(wmLoc)

    # check whether weather model files exists and/or or should be downloaded
    if Path(path_wm_crop).exists() and not force_download:
        logger.warning(
            'Processed weather model already exists, please remove it ("%s") if you want to download a new one.',
            path_wm_crop,
        )

    # check whether the raw weather model covers this area
    elif Path(path_wm_raw).exists() and checkContainment_raw(path_wm_raw, ll_bounds) and not force_download:
        logger.warning(
            'Raw weather model already exists, please remove it ("%s") if you want to download a new one.',
            path_wm_raw,
        )

    # if no weather model files supplied, check the standard location
    else:
        Path(path_wm_raw).parent.mkdir(parents=True, exist_ok=True)
        try:
            weather_model.fetch(Path(path_wm_raw), time)
        except DatetimeOutsideRange:
            raise TryToKeepGoingError

    # If only downloading, exit now
    if download_only:
        logger.warning('download_only flag selected. No further processing will happen.')
        return None

    # Otherwise, load the weather model data
    f = weather_model.load()

    if f is not None:
        logger.warning('The processed weather model file already exists, so I will use that.')

        containment = weather_model.checkContainment(ll_bounds)
        if not containment and weather_model.Model() not in ['HRRR', 'HRRRAK']:
            raise ExistingWeatherModelTooSmall

        return f

    # Logging some basic info
    logger.debug('Number of weather model nodes: %s', np.prod(weather_model.getWetRefractivity().shape))
    shape = weather_model.getWetRefractivity().shape
    logger.debug(f'Shape of weather model: {shape}')
    logger.debug(
        'Bounds of the weather model: %.2f/%.2f/%.2f/%.2f (SNWE)',
        np.nanmin(weather_model._ys),
        np.nanmax(weather_model._ys),
        np.nanmin(weather_model._xs),
        np.nanmax(weather_model._xs),
    )
    logger.debug('Weather model: %s', weather_model.Model())
    logger.debug('Mean value of the wet refractivity: %f', np.nanmean(weather_model.getWetRefractivity()))
    logger.debug('Mean value of the hydrostatic refractivity: %f', np.nanmean(weather_model.getHydroRefractivity()))
    logger.debug(weather_model)

    if makePlots:
        weather_model.plot('wh', True)
        weather_model.plot('pqt', True)
        plt.close('all')

    try:
        f = weather_model.write()
        containment = weather_model.checkContainment(ll_bounds)

    except Exception as e:
        logger.exception('Unable to save weathermodel to file')
        logger.exception(e)
        raise CriticalError from e

    finally:
        wm = weather_model.Model()
        del weather_model

    if not containment and wm not in ['HRRR']:
        raise ExistingWeatherModelTooSmall
    else:
        return f
