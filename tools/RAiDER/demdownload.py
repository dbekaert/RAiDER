#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import gdal
import numpy as np
import os
from scipy.interpolate import RegularGridInterpolator as rgi

import RAiDER.utilFcns

_world_dem = ('https://cloud.sdsc.edu/v1/AUTH_opentopography/Raster/'
              'SRTM_GL1_Ellip/SRTM_GL1_Ellip_srtm.vrt')


def download_dem(lats, lons, outLoc = None, save_flag= 'new', checkDEM = True, 
                  outName = 'warpedDEM.dem', ndv = 0.):
    '''
    Download a DEM if one is not already present. 
    '''
    print('Getting the DEM')

    # Insert check for DEM noData values
    if checkDEM:
        lats[lats==ndv] = np.nan
        lons[lons==ndv] = np.nan

    minlon = np.nanmin(lons) - 0.02
    maxlon = np.nanmax(lons) + 0.02
    minlat = np.nanmin(lats) - 0.02
    maxlat = np.nanmax(lats) + 0.02

    # Make sure the DEM hasn't already been downloaded
    if outLoc is not None:
        outRasterName = os.path.join(outLoc, outName) 
    else:
        outRasterName = outName
 
    if os.path.exists(outRasterName):
        print('WARNING: DEM already exists in {}, checking shape'.format(os.path.dirname(outRasterName)))
        try:
            hgts = RAiDER.utilFcns.gdal_open(outRasterName)
            if hgts.shape != lats.shape:
                raise RuntimeError('Existing DEM does not cover the area of the input \n \
                              lat/lon points; either move the DEM, delete it, or \n \
                              change the inputs.')
        except RuntimeError:
            hgts = RAiDER.utilFcns.read_hgt_file(outRasterName)
             
        hgts[hgts==ndv] = np.nan
        return hgts


    # Specify filenames
    memRaster = '/vsimem/warpedDEM'
    inRaster ='/vsicurl/{}'.format(_world_dem) 

    ds = gdal.Open(inRaster)
    gdalNDV = ds.GetRasterBand(1).GetNoDataValue()
    del ds

    # Download and warp
    print('Beginning DEM download and warping')
    
    wrpOpt = gdal.WarpOptions(outputBounds = (minlon, minlat,maxlon, maxlat))
    gdal.Warp(memRaster, inRaster, options = wrpOpt)

    print('DEM download finished')

    # Load the DEM data
    out = RAiDER.utilFcns.gdal_open(memRaster)

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

    print('Interpolation finished')

    if save_flag=='new':
        print('Saving DEM to disk')
        # ensure folders are created
        folderName = os.sep.join(os.path.split(outRasterName)[:-1])
        os.makedirs(folderName, exist_ok=True)

        # Need to ensure that noData values are consistently handled and 
        # can be passed on to GDAL
        outInterp[np.isnan(outInterp)] = ndv
        if outInterp.ndim==2:
            RAiDER.utilFcns.writeArrayToRaster(outInterp, outRasterName, noDataValue = gdalNDV)
        elif outInterp.ndim==1:
            RAiDER.utilFcns.writeArrayToFile(lons, lats, outInterp, outRasterName, noDataValue = ndv)
        else:
            raise RuntimeError('Why is the DEM 3-dimensional?')
    elif save_flag=='merge':
       import pandas as pd
       df = pd.read_csv(outRasterName)
       df['Hgt_m'] = outInterp
       df.to_csv(outRasterName)
    else:
       pass

    return outInterp
