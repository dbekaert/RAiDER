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

import RAiDER.utilFcns

from RAiDER.interpolator import interpolateDEM
from RAiDER.logger import *
from RAiDER.utilFcns import gdal_open, gdal_extents


_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def getHeights(lats, lons, heights, useWeatherNodes=False):
    '''
    Fcn to return heights from a DEM, either one that already exists
    or will download one if needed.
    '''
    height_type, height_data = heights
    in_shape = lats.shape

    if height_type == 'dem':
        try:
            hts = gdal_open(height_data)
        except:
            log.warning(
                'File %s could not be opened; requires GDAL-readable file.',
                height_data, exc_info=True
            )
            log.info('Proceeding with DEM download')
            height_type = 'download'

    elif height_type == 'lvs':
        if height_data is not None and useWeatherNodes:
            hts = height_data
        elif height_data is not None:
            hts = height_data
            latlist, lonlist, hgtlist = [], [], []
            for ht in hts:
                latlist.append(lats.flatten())
                lonlist.append(lons.flatten())
                hgtlist.append(np.array([ht] * len(lats.flatten())))
            lats = np.array(latlist).reshape(in_shape + (len(height_data),))
            lons = np.array(lonlist).reshape(in_shape + (len(height_data),))
            hts = np.array(hgtlist).reshape(in_shape + (len(height_data),))
        else:
            raise RuntimeError('Heights must be specified with height option "lvs"')

    elif height_type == 'merge':
        import pandas as pd
        for f in height_data:
            data = pd.read_csv(f)
            lats = data['Lat'].values
            lons = data['Lon'].values
            hts = download_dem(lats, lons, outName=f, save_flag='merge')
    else:
        if useWeatherNodes:
            hts = None
            height_type = 'skip'
        else:
            height_type = 'download'

    if height_type == 'download':
        hts = download_dem(lats, lons, outName=os.path.abspath(height_data))

    lats, lons, hts = [forceNDArray(v) for v in (lats, lons, hts)]

    return lats, lons, hts


def forceNDArray(arg):
    if arg is None:
        return None
    else:
        return np.array(arg)


def download_dem(
        lats, 
        lons, 
        save_flag='new', 
        checkDEM=True,
        outName=None, 
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

    inExtent = getBufferedExtent(lats, lons, buf=buf)

    # See if the DEM already exists
    if os.path.exists(outName):
        logger.warning(
            'A DEM already exists in {}, checking extents'
            .format(os.path.dirname(outName))
        )
        
        try:
            if isOutside(
                inExtent, 
                getBufferedExtent(
                    gdal_extents(outName), 
                    buf=buf
                )
            ):
                raise ValueError(
                    'Existing DEM does not cover the area of the input lat/lon '
                    'points; either move the DEM, delete it, or change the input '
                     'points.'
                )
            else:
                hgts = gdal_open(outName)
                hgts[hgts == ndv] = np.nan
                return hgts

        #TOFIX: Even if the DEM covers the input extent we may still need to interpolate it
        except AttributeError:
            logging.warning(
                'Existing DEM does not contain geo-referencing info, so ' 
                'I will download a new one.'
            )
        except OSError:
            hgts = RAiDER.utilFcns.read_hgt_file(outName)
            hgts[hgts == ndv] = np.nan
            return hgts
       

    # Otherwise download a global DEM
    logger.debug('Getting the DEM')
    out = getDEM(inExtent)
    logger.debug('Loaded the DEM')

    # Interpolate to the query points
    logger.debug('Beginning interpolation')
    outInterp = interpolateDEM(out, np.stack((lats, lons), axis=-1), inExtent)
    logger.debug('Interpolation finished')

    # save the DEM
    if outName is None:
        outName = os.path.join(os.getcwd(), 'geom', 'warpedDEM.dem')

    if save_flag == 'new':
        logger.debug('Saving DEM to disk')
        # ensure folders are created
        folderName = os.sep.join(os.path.split(outName)[:-1])
        os.makedirs(folderName, exist_ok=True)

        # Need to ensure that noData values are consistently handled and
        # can be passed on to GDAL
        outInterp[np.isnan(outInterp)] = ndv
        if outInterp.ndim == 2:
            RAiDER.utilFcns.writeArrayToRaster(outInterp, outName, noDataValue=ndv)
        elif outInterp.ndim == 1:
            RAiDER.utilFcns.writeArrayToFile(lons, lats, outInterp, outName, noDataValue=ndv)
        else:
            raise RuntimeError('Why is the DEM 3-dimensional?')
    elif save_flag == 'merge':
        import pandas as pd
        df = pd.read_csv(outName)
        df['Hgt_m'] = outInterp
        df.to_csv(outName, index=False)
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


def getDEM(extent, inRaster = '/vsicurl/{}'.format(_world_dem)):
    memRaster = '/vsimem/warpedDEM'
    gdal_extent = [extent[2], extent[0], extent[3], extent[1]]
    gdal.BuildVRT(memRaster, inRaster, outputBounds=gdal_extent)
    out = gdal_open(memRaster)
    out = out[::-1] # GDAL returns rasters upside-down
    return out


