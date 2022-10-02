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
import xarray

from netCDF4 import Dataset
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator as Interpolator

from RAiDER.constants import _STEP
from RAiDER.delayFcns import (
    getInterpolators,
    calculate_start_points,
    get_delays,
)
from RAiDER.dem import getHeights
from RAiDER.logger import logger
from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.processWM import prepareWeatherModel
from RAiDER.utilFcns import (
    gdal_open, writeDelays, projectDelays, writePnts2HDF5, lla2ecef,
)


def tropo_delay(args):
    """
    raiderDelay main function.
    
    Parameters
    ----------
    args: dict  Parameters and inputs needed for processing, 
                containing the following key-value pairs:
        
        los     - tuple, Zenith class object, ('los', 2-band los file), or ('sv', orbit_file)
        lats    - ndarray
        lons    - ndarray
        heights - see checkArgs for format
        flag    - 
        weather_model   - type of weather model to use
        wmLoc   - Directory containing weather model files
        zref    - max integration height
        outformat       - File format to use for raster outputs
        time    - list of datetimes to calculate delays
        download_only   - Only download the raw weather model data and exit
        wetFilename     - 
        hydroFilename   - 
        pnts_file       - Input a points file from previous run
        verbose - verbose printing
    """
    # unpacking the dictionairy
    los = args['los'] 
    lats = args['lats'] # 
    lons = args['lons']
    heights = args['heights']
    flag = args['flag']
    weather_model = args['weather_model']
    wmLoc = args['wmLoc']
    zref = args['zref']
    outformat = args['outformat']
    time = args['times']
    download_only = args['download_only']
    wetFilename = args['wetFilenames']
    hydroFilename = args['hydroFilenames']
    pnts_file = args['pnts_file']
    verbose = args['verbose']

    # logging
    logger.debug('Starting to run the weather model calculation')
    logger.debug('Time type: {}'.format(type(time)))
    logger.debug('Time: {}'.format(time.strftime('%Y%m%d')))
    logger.debug('Flag type is {}'.format(flag))
    logger.debug('DEM/height type is "{}"'.format(heights[0]))
    logger.debug('Max integration height is {:1.1f} m'.format(zref))

    ###########################################################
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
        makePlots=verbose,
    )

    if download_only:
        return None, None
    elif useWeatherNodes:
        if heights[0] == 'lvs':
            # compute delays at the correct levels
            ds = xarray.load_dataset(weather_model_file)
            ds['wet_total'] = ds['wet_total'].interp(z=heights[1])
            ds['hydro_total'] = ds['hydro_total'].interp(z=heights[1])
        else:
            logger.debug(
                'Only Zenith delays at the weather model nodes '
                'are requested, so I am exiting now. Delays have '
                'been written to the weather model file; see '
                '{}'.format(weather_model_file)
            )
        return None, None
    ###########################################################

    ###########################################################
    # If query points are specified, pull the height info
    logger.debug('Beginning DEM calculation')
    lats, lons, hgts = getHeights(lats, lons, heights, useWeatherNodes)
    logger.debug(
        'DEM height range for the queried region is %.2f-%.2f m',
        np.nanmin(hgts), np.nanmax(hgts)
    )

    # Transform the query points 
    pnt_proj = CRS.from_epsg(4326)
    ds = xarray.load_dataset(weather_model_file)
    try:
        wm_proj = ds['CRS']
    except:
        print("WARNING: I can't find a CRS in the weather model file, so I will assume you are using WGS84")
        wm_proj = 4326
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
    ####################################################################

    ####################################################################
    # Calculate delays
    los.setPoints(lats, lons, hgts)
    if (los is Zenith) or (los is Conventional):
        # either way I'll need the ZTD
        ifWet, ifHydro = getInterpolators(weather_model_file, 'total')
        wetDelay = ifWet(pnts)
        hydroDelay = ifHydro(pnts)

        # return the delays (ZTD or STD)
        wetDelay = los(wetDelay)
        hydroDelay = los(hydroDelay)

    else:
        ###########################################################
        # Full raytracing calculation
        # Requires handling the query points
        ###########################################################
        raise NotImplementedError

    del ds # cleanup

    ###########################################################
    # Write the delays to file
    # Different options depending on the inputs

    if heights[0] == 'lvs':
        outName = wetFilename[0].replace('wet', 'delays')
        writeDelays(
            flag,
            wetDelay,
            hydroDelay,
            lats,
            lons,
            outName,
            zlevels=hgts,
            outformat=outformat,
            delayType=delayType
        )
        logger.info('Finished writing data to %s', outName)

    else:
        if not isinstance(wetFilename, str):
            wetFilename = wetFilename[0]
            hydroFilename = hydroFilename[0]

        writeDelays(flag, wetDelay, hydroDelay, lats, lons,
                    wetFilename, hydroFilename, outformat=outformat,
                    proj=None, gt=None, ndv=0.)
        logger.info('Finished writing data to %s', wetFilename)

    return wetDelay, hydroDelay


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
