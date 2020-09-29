#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import logging
import os
import time

import numpy as np
from osgeo import gdal
from scipy.interpolate import RegularGridInterpolator as rgi

import RAiDER.utilFcns

from RAiDER.logger import *
from RAiDER.utilFcns import gdal_open, gdal_extents
#TODO: implement gdal_extents

_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def download_dem(
        lats, 
        lons, 
        outLoc=None, 
        save_flag='new', 
        checkDEM=True,
        outName='warpedDEM.dem', 
        ndv=0., 
        buf = 0.02
    ):
    '''
    Download a DEM if one is not already present.
    '''
    logger.debug('Getting the DEM')

    # Insert check for DEM noData values
    if checkDEM:
        lats[lats == ndv] = np.nan
        lons[lons == ndv] = np.nan

    minlat, maxlat, minlon, maxlon = getBufferedExtent(lats, lons, buf=buf)

    # Make sure the DEM hasn't already been downloaded
    if outLoc is not None:
        outRasterName = os.path.join(outLoc, outName)
    else:
        outRasterName = outName

    if os.path.exists(outRasterName):
        logger.warning(
            'DEM already exists in %s, checking extents',
            os.path.dirname(outRasterName)
        )
        
        if isOutside(
            [minlat, maxlat, minlon, maxlon], 
            getBufferedExtent(
                gdal_extents(outRasterName), 
                buf=buf
            )
        ):
            raise RuntimeError(
                'Existing DEM does not cover the area of the input lat/lon '
                'points; either move the DEM, delete it, or change the inputs.'
            )
        hgts[hgts == ndv] = np.nan
        return hgts


    # Pull the DEM
    logger.debug('Getting the DEM')
    out = getGlobalDEM()
    logger.debug('Loaded the DEM')
    logger.debug('Beginning interpolation')

    # Interpolate to the query points
    nPixLat = out.shape[0]
    nPixLon = out.shape[1]
    xlats = np.linspace(minlat, maxlat, nPixLat)
    xlons = np.linspace(minlon, maxlon, nPixLon)
    interpolator = rgi(points=(xlats, xlons), values=out,
                       method='linear',
                       bounds_error=False)
    outInterp = interpolator(np.stack((lats, lons), axis=-1))
    logger.debug('Interpolation finished')

    # save the DEM
    if save_flag == 'new':
        logger.debug('Saving DEM to disk')
        # ensure folders are created
        folderName = os.sep.join(os.path.split(outRasterName)[:-1])
        os.makedirs(folderName, exist_ok=True)

        # Need to ensure that noData values are consistently handled and
        # can be passed on to GDAL
        outInterp[np.isnan(outInterp)] = ndv
        if outInterp.ndim == 2:
            RAiDER.utilFcns.writeArrayToRaster(outInterp, outRasterName, noDataValue=ndv)
        elif outInterp.ndim == 1:
            RAiDER.utilFcns.writeArrayToFile(lons, lats, outInterp, outRasterName, noDataValue=ndv)
        else:
            raise RuntimeError('Why is the DEM 3-dimensional?')
    elif save_flag == 'merge':
        import pandas as pd
        df = pd.read_csv(outRasterName)
        df['Hgt_m'] = outInterp
        df.to_csv(outRasterName, index=False)
    else:
        pass

    return outInterp


def getBufferedExtent(lats, lons = None, buf = 0.):
    '''
    get the bounding box around a set of lats/lons
    '''
    if lons is None:
        lats, lons = lats[...,0], lons[...,1]
            
    try:
        if (lats.size == 1) & (lons.size == 1):
            out = [lats - buf, lats + buf, lons - buf, lons + buf]
        elif (lats.size > 1) & (lons.size > 1):
            out = [np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)]
        elif lats.size == 1:
            out = [lats - buf, lats + buf, np.nanmin(lons), np.nanmax(lons)]
        elif lons.size == 1:
            out = [np.nanmin(lats), np.nanmax(lats), lons - buf, lons + buf]
    except AttributeError:
        if isinstance(lats, tuple) and len(lats)==2:
            out = [min(lats) - buf, max(lats) + buf, min(lons) - buf, max(lons) + buf]
    except Exception as e:
        logger.warning('getBufferExtent failed: lats type: {}\n, content: {}'.format(type(lats), lats))
        logger.error(e)
        raise RuntimeError('Not a valid lat/lon shape or variable')

    return np.array(out)


def getGlobalDEM(inRaster = '/vsicurl/{}'.format(_world_dem)):
    memRaster = '/vsimem/warpedDEM'
    gdal.BuildVRT(memRaster, inRaster, outputBounds=[minlon, minlat, maxlon, maxlat])
    out = gdal_open(memRaster)
    out = out[::-1] # GDAL returns rasters upside-down
    return out


def isOutside(extent1, extent2):
    '''
    Determine whether any of extent1  lies outside extent2
    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon]
    Equal extents are considered "inside"
    '''
    t1 = extent1[0] < extent2[0]
    t2 = extent1[1] > extent2[1]
    t3 = extent1[2] < extent2[2]
    t4 = extent1[3] > extent2[3]
    if np.any([t1, t2, t3, t4]):
        return True
    return False


def isInside(extent1, extent2):
    '''
    Determine whether all of extent1 lies inside extent2
    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon].
    Equal extents are considered "inside"
    '''
    t1 = extent1[0] <= extent2[0]
    t2 = extent1[1] >= extent2[1]
    t3 = extent1[2] <= extent2[2]
    t4 = extent1[3] >= extent2[3]
    if np.all([t1, t2, t3, t4]):
        return True
    return False
