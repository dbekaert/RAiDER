# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import numpy as np
from RAiDER.logger import logger
from RAiDER.utilFcns import getTimeFromFile
import matplotlib.pyplot as plt


def main(
    weather_model,
    time=None,
    wmLoc=None,
    ll_bounds=None,
    zref=None,
    download_only=False,
    makePlots=False,
    force_download=False,
):
    '''
    Parse inputs to download and prepare a weather model grid for interpolation
    '''
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
            if not force_download:
                logger.warning(
                    'Weather model already exists, please remove it ("%s") if you want '
                    'to download a new one.', weather_model.files
                )
                download_flag = False
    else:
        download_flag = False

    # if no weather model files supplied, check the standard location
    if download_flag:
        weather_model.fetch(*weather_model.files, ll_bounds, time)
    else:
        time = getTimeFromFile(weather_model.files[0])
        weather_model.setTime(time)
        containment = weather_model.checkContainment(ll_bounds)

        if not containment:
            logger.error(
                'The weather model passed does not cover all of the input '
                'points; you need to download a larger area.'
            )
            raise RuntimeError(
                'The weather model passed does not cover all of the input '
                'points; please delete or rename it so another can be downloaded.'
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
        ll_bounds = ll_bounds,
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


def checkBounds(weather_model, outLats, outLons):
    '''Check the bounds of a weather model'''
    ds = xr.load_dataset(weather_model.files[0])  # TODO: xr is undefined
    coords = ds.coords  # coords is dict-like
    keys = [k for k in coords.keys()]
    xc = coords[keys[0]]
    yc = coords[keys[1]]
    lat_bounds = [yc.min(), yc.max()]
    lon_bounds = [xc.min(), xc.max()]
    self_extent = lat_bounds + lon_bounds
    in_extent = weather_model._getExtent(outLats, outLons)

    return in_extent, self_extent


def weather_model_debug(
    los,
    lats,
    lons,
    ll_bounds,
    weather_model,
    wmLoc,
    zref,
    time,
    out,
    download_only
):
    """
    raiderWeatherModelDebug main function.
    """

    logger.debug('Starting to run the weather model calculation with debugging plots')
    logger.debug('Time type: %s', type(time))
    logger.debug('Time: %s', time.strftime('%Y%m%d'))

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    # TODO: make_weather_model_filename is undefined
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
            zref=zref,
            download_only=download_only,
            makePlots=True
        )
        try:
            weather_model.write2NETCDF4(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")
