#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import glob
import numpy as np
import os
from osgeo import gdal
import scipy.interpolate
from scipy.interpolate import RegularGridInterpolator as rgi

import RAiDER.util
gdal.UseExceptions()


_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def download_dem(lats, lons, outLoc, save_flag= True, checkDEM = True, outName = 'warpedDEM.dem'):
    '''
    Download a DEM if one is not already present. 
    '''
    gdalNDV = 0
    print('Getting the DEM')

    # Insert check for DEM noData values
    if checkDEM:
        lats[lats==0.] = np.nan
        lons[lons==0.] = np.nan

    minlon = np.nanmin(lons) - 0.02
    maxlon = np.nanmax(lons) + 0.02
    minlat = np.nanmin(lats) - 0.02
    maxlat = np.nanmax(lats) + 0.02
    pixWidth = (maxlon - minlon)/lats.shape[1]
    pixHeight = (maxlat - minlat)/lats.shape[1]

    # Make sure the DEM hasn't already been downloaded
    outRasterName = os.path.join(outLoc, outName) 
 
    if os.path.exists(outRasterName):
       print('WARNING: DEM already exists in {}, checking shape'.format(outLoc))
       hgts = util.gdal_open(outRasterName)
       if hgts.shape != lats.shape:
          raise RuntimeError('Existing DEM does not cover the area of the input \n \
                              lat/lon points; either move the DEM, delete it, or \n \
                              change the inputs.')
       hgts[hgts==0.] = np.nan
       return hgts


    # Specify filenames
    memRaster = '/vsimem/warpedDEM'
    inRaster ='/vsicurl/{}'.format(_world_dem) 

    # Download and warp
    print('Beginning DEM download and warping')
    
    wrpOpt = gdal.WarpOptions(outputBounds = (minlon, minlat,maxlon, maxlat))
    gdal.Warp(memRaster, inRaster, options = wrpOpt)

    print('DEM download finished')

    # Load the DEM data
    try:
        out = util.gdal_open(memRaster)
    except:
        raise RuntimeError('demdownload: Cannot open the warped file')
    finally:
        try:
            gdal.Unlink('/vsimem/warpedDEM')
        except:
            pass

    #  Flip the orientation, since GDAL writes top-bot
    out = out[::-1]

    print('Beginning interpolation')
    nPixLat = out.shape[0]
    nPixLon = out.shape[1]
    xlats = np.linspace(minlat, maxlat, nPixLat)
    xlons = np.linspace(minlon, maxlon, nPixLon)
    interpolator = rgi(points = (xlats, xlons),values = out,
                       method='linear', 
                       bounds_error = False)

    outInterp = interpolator(np.stack((lats, lons), axis=-1))

    # Need to ensure that noData values are consistently handled and 
    # can be passed on to GDAL
    outInterp[np.isnan(outInterp)] = gdalNDV
    outInterp[outInterp < -10000] = gdalNDV

    print('Interpolation finished')

    if save_flag:
        print('Saving DEM to disk')
        if outInterp.ndim==2:

            util.writeArrayToRaster(outInterp, outRasterName, noDataValue = gdalNDV)

        elif outInterp.ndim==1:
            util.writeArrayToFile(lons, lats, outInterp, outRasterName, noDataValue = -9999)
        else:
            raise RuntimeError('Why is the DEM 3-dimensional?')
        print('Finished saving DEM to disk')

    return outInterp
