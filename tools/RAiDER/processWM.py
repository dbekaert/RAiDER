# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import matplotlib.pyplot as plt
import numpy as np

from RAiDER.logger import logger
from RAiDER.models.customExceptions import (
    CriticalError,
    DatetimeOutsideRange,
    ExistingWeatherModelTooSmall,
    TryToKeepGoingError,
)
from RAiDER.models.weatherModel import checkContainment_raw, make_raw_weather_data_filename, make_weather_model_filename


def prepareWeatherModel(
    weather_model,
    time,
    ll_bounds,
    download_only: bool=False,
    makePlots: bool=False,
    force_download: bool=False,
) -> str:
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
    ## set the bounding box from the in the case that it hasn't been set
    if weather_model.get_latlon_bounds() is None:
        weather_model.set_latlon_bounds(ll_bounds)

    # Ensure the file output location exists
    wmLoc     = weather_model.get_wmLoc()
    weather_model.setTime(time)

    # get the path to the less processed weather model file
    path_wm_raw = make_raw_weather_data_filename(wmLoc, weather_model.Model(), time)

    # get the path to the more processed (cropped) weather model file
    path_wm_crop = weather_model.out_file(wmLoc)

    # check whether weather model files exists and/or or should be downloaded
    if os.path.exists(path_wm_crop) and not force_download:
        logger.warning(
            'Processed weather model already exists, please remove it ("%s") if you want '
            'to download a new one.', path_wm_crop)

    # check whether the raw weather model covers this area
    elif os.path.exists(path_wm_raw) and \
        checkContainment_raw(path_wm_raw, ll_bounds) and not force_download:
        logger.warning(
            'Raw weather model already exists, please remove it ("%s") if you want '
            'to download a new one.', path_wm_raw)

    # if no weather model files supplied, check the standard location
    else:
        os.makedirs(os.path.dirname(path_wm_raw), exist_ok=True)
        try:
            weather_model.fetch(path_wm_raw, time)
        except DatetimeOutsideRange:
            raise TryToKeepGoingError

    # If only downloading, exit now
    if download_only:
        logger.warning(
            'download_only flag selected. No further processing will happen.'
        )
        return None

    # Otherwise, load the weather model data
    f = weather_model.load()

    if f is not None:
        logger.warning(
            'The processed weather model file already exists,'
            ' so I will use that.'
        )

        containment = weather_model.checkContainment(ll_bounds)
        if not containment and weather_model.Model() not in 'HRRR'.split():
            raise ExistingWeatherModelTooSmall

        return f

    # Logging some basic info
    logger.debug(
        'Number of weather model nodes: %s',
            np.prod(weather_model.getWetRefractivity().shape)
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
        containment = weather_model.checkContainment(ll_bounds)

    except Exception as e:
        logger.exception("Unable to save weathermodel to file")
        logger.exception(e)
        raise CriticalError

    finally:
        wm = weather_model.Model()
        del weather_model

    if not containment and wm not in 'HRRR'.split():
        raise ExistingWeatherModelTooSmall
    else:
        return f


def _weather_model_debug(
    los,
    lats,
    lons,
    ll_bounds,
    weather_model,
    wmLoc,
    time,
    out,
    download_only
) -> None:
    """RaiderWeatherModelDebug main function."""
    logger.debug('Starting to run the weather model calculation with debugging plots')
    logger.debug('Time type: %s', type(time))
    logger.debug('Time: %s', time.strftime('%Y%m%d'))

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = make_weather_model_filename(
        weather_model['name'],
        time,
        ll_bounds
    )
    weather_model_file = os.path.join(wmLoc, wm_filename)

    if not os.path.exists(weather_model_file):
        prepareWeatherModel(
            weather_model,
            time,
            wmLoc=wmLoc,
            lats=lats,
            lons=lons,
            ll_bounds=ll_bounds,
            download_only=download_only,
            makePlots=True
        )
        try:
            weather_model.write2NETCDF4(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")
