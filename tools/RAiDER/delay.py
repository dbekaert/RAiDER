#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import h5py
import numpy as np

import RAiDER.delayFcns
from RAiDER.constants import _STEP, _ZREF, Zenith
from RAiDER.interpolator import interp_along_axis
from RAiDER.dem import getHeights
from RAiDER.logger import *
from RAiDER.losreader import getLookVectors
from RAiDER.processWM import prepareWeatherModel
from RAiDER.rays import getLookVectors
from RAiDER.utilFcns import (
    make_weather_model_filename, writeDelays, writePnts2HDF5
)


def interpolateDelay(weather_model_file_name, pnts_file_name,
                     zlevels=None, zref=_ZREF, stepSize=_STEP,
                     interpType='rgi', nproc=8,
                     useDask=False, delayType="Zenith"):
    """
    This function calculates the line-of-sight vectors, estimates the point-wise refractivity
    index for each one, and then integrates to get the total delay in meters. The point-wise
    delay is calculated by interpolating the weatherObj, which contains a weather model with
    wet and hydrostatic refractivity at each weather model grid node, to the points along
    the ray. The refractivity is integrated along the ray to get the final delay.

    Inputs:
     weatherObj - a weather model object
     heights    - Grid of heights for each ground point
     look_vecs  - Grid of look vectors streching from ground point to sensor (cut off at zref)
     stepSize   - Integration step size in meters
     intpType   - Can be one of 'scipy': LinearNDInterpolator, or 'sane': _sane_interpolate.
                  Any other string will use the RegularGridInterpolate method
     nproc      - Number of parallel processes to use if useDask is True
     useDask    - use Dask to parallelize ray calculation

    Outputs:
     delays     - A list containing the wet and hydrostatic delays for each ground point in
                  meters.
    """
    logger.debug('Beginning ray calculation')
    logger.debug('ZREF = %s', zref)
    logger.debug('stepSize = %f', stepSize)

    RAiDER.delayFcns.calculate_rays(pnts_file_name, stepSize)
    return RAiDER.delayFcns.get_delays(
        stepSize, pnts_file_name, weather_model_file_name,
        interpType=interpType, delayType=delayType
    )


def computeDelay(weather_model_file_name, pnts_file_name, useWeatherNodes=False,
                 zlevels=None, zref=_ZREF, out=None, parallel=False,
                 delayType="Zenith"):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    interpolateDelay.
    """
    logger.debug('Beginning delay calculation')

    if parallel:
        useDask = True
        nproc = 16
    else:
        useDask = False
        nproc = 1

    logger.debug('Reference z-value (max z for integration) is %s m', zref)
    logger.debug('Number of processors to use: %d', nproc)

    # If weather model nodes only are desired, the calculation is very quick
    if useWeatherNodes:
        # Get the weather model data
        with h5py.File(weather_model_file_name, 'r') as f:
            zs_wm = f['z'][()].copy()
            total_wet = f['wet_total'][()].copy()
            total_hydro = f['hydro_total'][()].copy()
        if zlevels is None:
            return total_wet, total_hydro
        else:
            wet_delays = interp_along_axis(zs_wm, zlevels, total_wet, axis=-1)
            hydro_delays = interp_along_axis(zs_wm, zlevels, total_hydro, axis=-1)
            return wet_delays, hydro_delays
    else:
        wet, hydro = interpolateDelay(weather_model_file_name, pnts_file_name, zlevels=zlevels,
                                      zref=zref, nproc=nproc, useDask=useDask,
                                      delayType=delayType)
        logger.debug('Finished delay calculation')

        return wet, hydro


def tropo_delay(los, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref,
                outformat, time, out, download_only, wetFilename, hydroFilename):
    """
    raiderDelay main function.
    """

    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: %s', type(time))
    logger.debug('Time: %s', time.strftime('%Y%m%d'))
    logger.debug('Flag type is %s', flag)
    logger.debug('DEM/height type is "%s"', heights[0])

    # Flags
    useWeatherNodes = flag == 'bounding_box'
    delayType = ["Zenith" if los is Zenith else "LOS"]

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = make_weather_model_filename(weather_model['name'], time, ll_bounds)
    weather_model_file = os.path.join(wmLoc, wm_filename)
    if not os.path.exists(weather_model_file):
        weather_model, lats, lons = prepareWeatherModel(
            weather_model, wmLoc, out, lats=lats, lons=lons, los=los, zref=zref,
            time=time, download_only=download_only, makePlots=True
        )
        try:
            weather_model.write2HDF5(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")

        del weather_model
    else:
        logger.warning(
            'Weather model already exists, please remove it ("%s") if you want '
            'to create a new one.', weather_model_file
        )

    if download_only:
        return None, None

    # Pull the DEM.
    logger.debug('Beginning DEM calculation')
    in_shape = lats.shape
    lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)

    pnts_file = None
    if not useWeatherNodes:
        pnts_file = os.path.join(out, 'geom', 'query_points.h5')
        if not os.path.exists(pnts_file):

            # Convert the line-of-sight inputs to look vectors
            logger.debug('Lats shape is %s', lats.shape)
            logger.debug(
                'lat/lon box is %f/%f/%f/%f (SNWE)',
                np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)
            )
            logger.debug(
                'DEM height range is %.2f-%.2f m',
                np.nanmin(hgts), np.nanmax(hgts)
            )
            logger.debug('Beginning line-of-sight calculation')
            los = getLookVectors(los, lats, lons, hgts, zref)

            # write to an HDF5 file
            writePnts2HDF5(lats, lons, hgts, los, outName=pnts_file)

    wetDelay, hydroDelay = computeDelay(
        weather_model_file, pnts_file, useWeatherNodes, zref, out,
        delayType=delayType
    )

    if heights[0] == 'lvs':
        outName = wetFilename.replace('wet', 'delays')
        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    outName, zlevels=hgts, outformat=outformat, delayType=delayType)
        logger.info('Finished writing data to %s', outName)
    elif useWeatherNodes:
        logger.info(
            'Delays have been written to the weather model file; see %s',
            weather_model_file
        )
    else:
        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        logger.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay


def weather_model_debug(los, lats, lons, ll_bounds, weather_model, wmLoc, zref,
                        time, out, download_only):
    """
    raiderWeatherModelDebug main function.
    """

    log.debug('Starting to run the weather model calculation with debugging plots')
    log.debug('Time type: %s', type(time))
    log.debug('Time: %s', time.strftime('%Y%m%d'))

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = make_weather_model_filename(weather_model['name'], time, ll_bounds)
    weather_model_file = os.path.join(wmLoc, wm_filename)
    if not os.path.exists(weather_model_file):
        weather_model, lats, lons = prepareWeatherModel(
            weather_model, wmLoc, out, lats=lats, lons=lons, los=los, zref=zref,
            time=time, download_only=download_only, makePlots=True
        )
        try:
            weather_model.write2HDF5(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")

        del weather_model
    else:
        logger.warning(
            'Weather model already exists, please remove it ("%s") if you want '
            'to create a new one.', weather_model_file
        )

    return 1
