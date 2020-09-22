#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import logging
import os

import h5py
import numpy as np

import RAiDER.delayFcns

from RAiDER.constants import _STEP, _ZREF, Zenith
from RAiDER.interpolator import interp_along_axis
from RAiDER.llreader import getHeights
from RAiDER.logger import *
from RAiDER.processWM import prepareWeatherModel
from RAiDER.rays import getLookVectors
from RAiDER.utilFcns import (
    make_weather_model_filename, writeDelays, writePnts2HDF5
)

def tropo_delay(losGen, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref,
                outformat, time, out, download_only, wetFilename, hydroFilename):
    """
    raiderDelay main function.
    """
    # Parameters
    stepSize = _STEP

    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: %s', type(time))
    logger.debug('Time: %s', time.strftime('%Y%m%d'))
    logger.debug('Flag type is %s', flag)
    logger.debug('DEM/height type is "%s"', heights[0])

    # Flags
    useWeatherNodes = flag == 'bounding_box'

    # location of the weather model files
    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = make_weather_model_filename(weather_model.Model(), time, ll_bounds)
    weather_model_file = os.path.join(wmLoc, wm_filename)
    if not os.path.exists(weather_model_file):
        weather_model, lats, lons = prepareWeatherModel(
            weather_model, wmLoc, lats=lats, lons=lons, los=losGen, zref=zref,
            time=time, download_only=download_only
        )
        try:
            weather_model.write2HDF5(weather_model_file)
        except Exception as e:
            logger.warning(e)
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

            logger.debug('Beginning line-of-sight calculation')
            los = getLookVectors(losGen, np.stack((lats, lons, hgts), axis=-1))

            # write to an HDF5 file
            writePnts2HDF5(lats, lons, hgts, los, outName=pnts_file)

    logger.debug('Beginning delay calculation')
    logger.debug('Reference z-value (max z for integration) is %f m', zref)
    logger.debug('Beginning ray calculation')
    logger.debug('stepSize = %f', stepSize)

    if useWeatherNodes:
        # If weather model nodes only are desired, the calculation is very quick
        with h5py.File(weather_model_file, 'r') as f:
            zs_wm = f['z'][()].copy()
            wet_delays = f['wet_total'][()].copy()
            hydro_delays = f['hydro_total'][()].copy()
        if heights[0] == 'lvs':
            zlevels = heights[1]
            wet_delays = interp_along_axis(zs_wm, zlevels, wet_delays, axis=-1)
            hydro_delays = interp_along_axis(zs_wm, zlevels, hydro_delays, axis=-1)
    else:
        wet_delays, hydro_delays = RAiDER.delayFcns.get_delays(
            stepSize, 
            pnts_file, 
            weather_model_file,
            zref=zref
        )

    logger.debug('Finished delay calculation')

    if heights[0] == 'lvs':
        outName = wetFilename.replace('wet', 'delays')
        writeDelays(flag, wet_delays, hydro_delays, lats, lons,
                    outName, zlevels=heights[1], outformat='hdf5', delayType=losGen.getLOSType())
        logger.info('Finished writing data to %s', outName)
    elif useWeatherNodes:
        logger.info(
            'Delays have been written to the weather model file; see %s',
            weather_model_file
        )
    else:
        writeDelays(flag, wet_delays, hydro_delays, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        logger.info('Finished writing data to %s', wetFilename)

    return wet_delays, hydro_delays
