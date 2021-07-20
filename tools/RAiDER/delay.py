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
from netCDF4 import Dataset

import RAiDER.delayFcns
from RAiDER.constants import _STEP, _ZREF, Zenith
from RAiDER.interpolator import interp_along_axis
from RAiDER.dem import getHeights
from RAiDER.logger import *
from RAiDER.losreader import getLookVectors
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import (
    writeDelays, writePnts2HDF5
)


def computeDelay(
    weather_model_file_name,
    pnts_file_name,
    useWeatherNodes=False,
    zlevels=None,
    zref=_ZREF,
    step=_STEP,
    out=None,
):
    """
    Calculate troposphere delay using a weather model file and query
    points file.
    """
    logger.debug('Beginning delay calculation')
    logger.debug('Max integration height is {:1.1f} m'.format(zref))
    logger.debug('Reference integration step is {:1.1f} m'.format(step))

    # If weather model nodes only are desired, the calculation is very quick
    if useWeatherNodes:
        # Get the weather model data
        with Dataset(weather_model_file_name, mode='r') as f:
            zs_wm = np.array(f.variables['z'][:])
            total_wet = np.array(f.variables['wet_total'][:]).swapaxes(1, 2).swapaxes(0, 2)
            total_hydro = np.array(f.variables['hydro_total'][:]).swapaxes(1, 2).swapaxes(0, 2)

        if zlevels is None:
            return total_wet, total_hydro
        else:
            wet_delays = interp_along_axis(zs_wm, zlevels, total_wet, axis=-1)
            hydro_delays = interp_along_axis(zs_wm, zlevels, total_hydro, axis=-1)
            return wet_delays, hydro_delays

    else:
        RAiDER.delayFcns.calculate_rays(
            pnts_file_name,
            step
        )


        wet, hydro = RAiDER.delayFcns.get_delays(
            step,
            pnts_file_name,
            weather_model_file_name,
        )

        logger.debug('Finished delay calculation')

        return wet, hydro


def tropo_delay(args):
    """
    raiderDelay main function.
    """

    # unpacking the dictionairy
    los = args['los']
    lats = args['lats']
    lons = args['lons']
    ll_bounds = args['ll_bounds']
    heights = args['heights']
    flag = args['flag']
    weather_model = args['weather_model']
    wmLoc = args['wmLoc']
    zref = args['zref']
    outformat = args['outformat']
    time = args['times']
    out = args['out']
    download_only = args['download_only']
    wetFilename = args['wetFilenames']
    hydroFilename = args['hydroFilenames']
    pnts_file = args['pnts_file']

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: {}'.format(type(time)))
    logger.debug('Time: {}'.format(time.strftime('%Y%m%d')))
    logger.debug('Flag type is {}'.format(flag))
    logger.debug('DEM/height type is "{}"'.format(heights[0]))

    # weather model calculation
    useWeatherNodes = flag == 'bounding_box'
    delayType = ["Zenith" if los is Zenith else "LOS"]

    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is {}'.format(download_only))

    weather_model_file = prepareWeatherModel(
        weather_model,
        time,
        wmLoc=wmLoc,
        lats=lats,
        lons=lons,
        zref=zref,
        download_only=download_only,
        makePlots=True
    )

    if download_only:
        return None, None

    if (los is Zenith) and (heights[0] == 'skip'):
        logger.debug('Only Zenith delays at the weather model nodes '
                     'are requested, so I am exiting now. ')
        return None, None

    # Pull the DEM.
    logger.debug('Beginning DEM calculation')
    lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)
    logger.debug(
        'DEM height range for the queried region is %.2f-%.2f m',
        np.nanmin(hgts), np.nanmax(hgts)
    )

    if heights[0] == 'lvs':
        zlevels = hgts
    else:
        zlevels = None

    query_shape = lats.shape
    logger.debug('Lats shape is {}'.format(query_shape))
    logger.debug(
        'lat/lon box is %f/%f/%f/%f (SNWE)',
        np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)
    )

    # Write the input query points to a file
    # Check whether the query points file already exists
    write_flag = checkQueryPntsFile(pnts_file, query_shape)

    # Throw an error if the user passes the same filename but different points
    if os.path.exists(pnts_file) and write_flag:
        logger.error(
            'The input query points file exists but does not match the '
            'shape of the input query points, either change the file '
            'name or delete the query points file ({})'.format(pnts_file)
        )
        raise ValueError(
            'The input query points file exists but does not match the '
            'shape of the input query points, either change the file '
            'name or delete the query points file ({})'.format(pnts_file)
        )

    if write_flag:
        logger.debug('Beginning line-of-sight calculation')

        # Convert the line-of-sight inputs to look vectors
        los = getLookVectors(los, lats, lons, hgts, time)
        import RAiDER.utilFcns as utilFcns
        los_enu = utilFcns.ecef2enu(los, lats, lons, hgts)
        np.save('./HR_SV/LOS_ENU.npy', los_enu)

        # write to an HDF5 file
        writePnts2HDF5(lats, lons, hgts, los, outName=pnts_file)

    else:
        logger.warning(
            'The input query points file already exists and matches the '
            'shape of the input query points, so I will use it.'
        )

    # Compute the delays
    wetDelay, hydroDelay = computeDelay(
        weather_model_file,
        pnts_file,
        useWeatherNodes,
        zlevels,
        zref,
        out=out,
    )

    if heights[0] == 'lvs':
        outName = wetFilename[0].replace('wet', 'delays')
        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    outName, zlevels=hgts, outformat=outformat, delayType=delayType)
        logger.info('Finished writing data to %s', outName)

    elif useWeatherNodes:
        logger.info(
            'Delays have been written to the weather model file; see %s',
            weather_model_file
        )

    else:
        if not isinstance(wetFilename, str):
            wetFilename = wetFilename[0]
            hydroFilename = hydroFilename[0]

        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        logger.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay


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

    log.debug('Starting to run the weather model calculation with debugging plots')
    log.debug('Time type: %s', type(time))
    log.debug('Time: %s', time.strftime('%Y%m%d'))

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
            zref=zref,
            download_only=download_only,
            makePlots=True
        )
        try:
            weather_model.write2NETCDF4(weather_model_file)
        except Exception:
            logger.exception("Unable to save weathermodel to file")

        del weather_model
    else:
        logger.warning(
            'Weather model already exists, please remove it ("%s") if you want '
            'to create a new one.', weather_model_file
        )

    return 1


def checkQueryPntsFile(pnts_file, query_shape):
    '''
    Check whether the query points file exists, and if it
    does, check that the shapes are all consistent
    '''
    write_flag = True
    if os.path.exists(pnts_file):
        # Check whether the number of points is consistent with the new inputs
        with h5py.File(pnts_file, 'r') as f:
            if query_shape == tuple(f['lon'].attrs['Shape']):
                write_flag = False

    return write_flag
