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
from RAiDER.processWM import prepareWeatherModel
from RAiDER.rays import getLookVectors
from RAiDER.utilFcns import (
    make_weather_model_filename, writeDelays, writePnts2HDF5
)

log = logging.getLogger(__name__)


def interpolateDelay(weather_model_file_name, pnts_file_name,
                     zlevels=None, zref=_ZREF, stepSize=_STEP,
                     interpType='rgi', nproc=8,
                     useDask=False):
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
    log.debug('Beginning ray calculation')
    log.debug('ZREF = %f', zref)
    log.debug('stepSize = %f', stepSize)

    RAiDER.delayFcns.calculate_rays(pnts_file_name, stepSize)
    return RAiDER.delayFcns.get_delays(
        stepSize, pnts_file_name, weather_model_file_name,
        interpType=interpType, zref=zref
    )


def computeDelay(weather_model_file_name, pnts_file_name, useWeatherNodes=False,
                 zlevels=None, zref=_ZREF, out=None, parallel=False):
    """Calculate troposphere delay from command-line arguments.

    We do a little bit of preprocessing, then call
    interpolateDelay.
    """
    log.debug('Beginning delay calculation')

    if parallel:
        useDask = True
        nproc = 16
    else:
        useDask = False
        nproc = 1

    log.debug('Reference z-value (max z for integration) is %f m', zref)
    log.debug('Number of processors to use: %d', nproc)

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
                                      zref=zref, nproc=nproc, useDask=useDask)
        log.debug('Finished delay calculation')

        return wet, hydro


def tropo_delay(losGen, lats, lons, ll_bounds, heights, flag, weather_model, wmLoc, zref,
                outformat, time, out, download_only, wetFilename, hydroFilename):
    """
    raiderDelay main function.
    """

    log.debug('Starting to run the weather model calculation')
    log.debug('Time type: %s', type(time))
    log.debug('Time: %s', time.strftime('%Y%m%d'))
    log.debug('Flag type is %s', flag)
    log.debug('DEM/height type is "%s"', heights[0])

    # Flags
    useWeatherNodes = flag == 'bounding_box'

    # location of the weather model files
    log.debug('Beginning weather model pre-processing')
    log.debug('Download-only is %s', download_only)
    if wmLoc is None:
        wmLoc = os.path.join(out, 'weather_files')

    # weather model calculation
    wm_filename = make_weather_model_filename(weather_model['name'], time, ll_bounds)
    weather_model_file = os.path.join(wmLoc, wm_filename)
    if not os.path.exists(weather_model_file):
        weather_model, lats, lons = prepareWeatherModel(
            weather_model, wmLoc, out, lats=lats, lons=lons, los=losGen, zref=zref,
            time=time, download_only=download_only
        )
        try:
            weather_model.write2HDF5(weather_model_file)
        except Exception:
            log.exception("Unable to save weathermodel to file")

        del weather_model
    else:
        log.warning(
            'Weather model already exists, please remove it ("%s") if you want '
            'to create a new one.', weather_model_file
        )

    if download_only:
        return None, None

    # Pull the DEM.
    log.debug('Beginning DEM calculation')
    in_shape = lats.shape
    lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)

    pnts_file = None
    if not useWeatherNodes:
        pnts_file = os.path.join(out, 'geom', 'query_points.h5')
        if not os.path.exists(pnts_file):

            # Convert the line-of-sight inputs to look vectors
            log.debug('Lats shape is %s', lats.shape)
            log.debug(
                'lat/lon box is %f/%f/%f/%f (SNWE)',
                np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)
            )
            log.debug(
                'DEM height range is %.2f-%.2f m',
                np.nanmin(hgts), np.nanmax(hgts)
            )
            log.debug('Beginning line-of-sight calculation')
            los = getLookVectors(losGen, np.stack((lats, lons, hgts), axis=-1), zref)

            # write to an HDF5 file
            writePnts2HDF5(lats, lons, hgts, los, outName=pnts_file)

    wetDelay, hydroDelay = computeDelay(
        weather_model_file_name=weather_model_file,
        pnts_file_name=pnts_file,
        useWeatherNodes=useWeatherNodes,
        zref=zref,
        out=out
    )

    if heights[0] == 'lvs':
        outName = wetFilename.replace('wet', 'delays')
        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    outName, zlevels=hgts, outformat=outformat, delayType=losGen.getLOSType())
        log.info('Finished writing data to %s', outName)
    elif useWeatherNodes:
        log.info(
            'Delays have been written to the weather model file; see %s',
            weather_model_file
        )
    else:
        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        log.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay
