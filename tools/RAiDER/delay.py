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
from pyproj import CRS, Transformer

from RAiDER.constants import _STEP
from RAiDER.delayFcns import (
    getInterpolators, calculate_rays, get_delays, getProjFromWMFile
)
from RAiDER.dem import getHeights
from RAiDER.logger import logger
from RAiDER.losreader import getLookVectors, Zenith
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import (
    gdal_open, writeDelays, projectDelays, writePnts2HDF5
)


def tropo_delay(args):
    """
    raiderDelay main function.
    """

    # unpacking the dictionary
    los = args['los']
    lats = args['lats']
    lons = args['lons']
    heights = args['heights']
    flag = args['flag']
    weather_model = args['weather_model']
    wmLoc = args['wmLoc']
    zref = args['zref']
    outformat = args['outformat']
    times = args['times']
    download_only = args['download_only']
    wetFilenames = args['wetFilenames']
    hydroFilenames = args['hydroFilenames']
    pnts_file = args['pnts_file']
    verbose = args['verbose']

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Times type: {}'.format(type(times)))
    logger.debug('Times: %s', times[0].strftime('%Y%m%d') if len(times) == 1 else times[0].strftime('%Y%m%d') + '/' + times[-1].strftime('%Y%m%d'))
    logger.debug('Flag type is {}'.format(flag))
    logger.debug('DEM/height type is "{}"'.format(heights[0]))
    logger.debug('Max integration height is {:1.1f} m'.format(zref))

    ###########################################################
    # weather model calculation
    useWeatherNodes = flag == 'bounding_box'
    delayType = ["Zenith" if los is Zenith else "LOS"]

    logger.debug('Beginning weather model pre-processing')
    logger.debug('Download-only is {}'.format(download_only))

    weather_model_files = prepareWeatherModel(
        weather_model,
        times,
        wmLoc=wmLoc,
        lats=lats,
        lons=lons,
        zref=zref,
        download_only=download_only,
        makePlots=verbose,
    )

    if useWeatherNodes:
        if heights[0] == 'lvs':
            # compute delays at the correct levels
            raise NotImplementedError
        else:
            logger.debug(
                'Only Zenith delays at the weather model nodes '
                'are requested, so I am exiting now. Delays have '
                'been written to the weather model file(s); see:\n'
                + '\n  '.join(weather_model_files)
            )
        return None, None

    ###########################################################
    # If query points are specified, pull the height info
    logger.debug('Beginning DEM calculation')
    lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)
    logger.debug(
        'DEM height range for the queried region is %.2f-%.2f m',
        np.nanmin(hgts), np.nanmax(hgts)
    )

    # Do different things if ZTD or STD is requested
    wetDelays = []
    hydroDelays = []
    for i, filename in enumerate(weather_model_files):
        if (los is Zenith) or (los[i] == 'los'):
            # Transform the query points if needed
            pnt_proj = CRS.from_epsg(4326)
            wm_proj = getProjFromWMFile(filename).to_string()
            if wm_proj != pnt_proj:
                pnts = transformPoints(
                    lats,
                    lons,
                    hgts,
                    pnt_proj,
                    wm_proj
                )
            else:
                # interpolators require y, x, z
                pnts = np.stack([lats, lons, hgts], axis=-1)

            # either way I'll need the ZTD
            ifWet, ifHydro = getInterpolators(filename, 'total')
            wetDelays.append(ifWet(pnts))
            hydroDelays.append(ifHydro(pnts))

            # Now do the projection if Conventional slant delay is requested
            if los is not Zenith:
                # inc, hd = gdal_open(los[1])
                inc = gdal_open(los[1])[i]
                wetDelays[i] = projectDelays(wetDelays[i], inc)
                hydroDelays[i] = projectDelays(hydroDelays[i], inc)

        else:
            ###########################################################
            # If asking for line-of-sight, do the full raytracing calculation
            # Requires handling the query points
            ###########################################################
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
                    'name or delete the query points file %s', pnts_file
                )

            if write_flag:
                logger.debug('Beginning line-of-sight calculation')

                # Convert the line-of-sight inputs to look vectors
                # TODO: time is undefined
                los, lengths = getLookVectors(los, lats, lons, hgts, zref=zref, time=time)

                # write to an HDF5 file
                writePnts2HDF5(lats, lons, hgts, los, lengths, out_name=pnts_file)

            logger.debug('Beginning raytracing calculation')
            logger.debug('Reference integration step is {:1.1f} m'.format(_STEP))

            calculate_rays(pnts_file, _STEP)

            # TODO: step is undefined
            wet, hydro = get_delays(step, pnts_file, filename)

            logger.debug('Finished raytracing calculation')

        ###########################################################
        # Write the delays to file
        # Different options depending on the inputs
        if heights[i] == 'lvs':
            out_name = wetFilenames[i].replace('wet', 'delays')
            writeDelays(
                flag,
                wetDelays[i],
                hydroDelays[i],
                lats,
                lons,
                out_name,
                zlevels=hgts,
                outformat=outformat,
                delayType=delayType
            )
            logger.info('Finished writing data to %s', out_name)

        else:
            writeDelays(flag, wetDelays[i], hydroDelays[i], lats, lons,
                        wetFilenames[i], hydroFilenames[i], outformat=outformat,
                        proj=None, gt=None, ndv=0.)
            logger.info('Finished writing data to %s', wetFilenames[i])

    return wetDelays, hydroDelays


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


def transformPoints(lats, lons, hgts, old_proj, new_proj):
    '''
    Transform lat/lon/hgt data to an array of points in a new
    projection

    Parameters
    ----------
    lats - WGS-84 latitude (EPSG: 4326)
    lons - ditto for longitude
    hgts - Ellipsoidal height in meters
    old_proj - the original projection of the points
    new_proj - the new projection in which to return the points

    Returns
    -------
    the array of query points in the weather model coordinate system
    '''
    t = Transformer.from_crs(old_proj, new_proj)
    return np.stack(t.transform(lats, lons, hgts), axis=-1).T
