# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from logging import warn
import os

import numpy as np
import pandas as pd

import rasterio
from dem_stitcher.stitcher import stitch_dem

import RAiDER.utilFcns

from RAiDER.interpolator import interpolateDEM
from RAiDER.logger import logger
from RAiDER.utilFcns import rio_open, rio_profile, rio_extents, get_file_and_band


def getHeights(ll_bounds, dem_type, dem_file, lats=None, lons=None):
    '''
    Fcn to return heights from a DEM, either one that already exists
    or will download one if needed.
    '''
    # Make sure lats/lons are passed if needed
    if ((dem_type == 'download') or (dem_type == 'dem')) and (lats is None):
        raise RuntimeError('lats/lons must be specified to interpolate from a DEM')

    # height_type, height_data = heights
    if dem_type == 'hgt':
        htinfo = get_file_and_band(dem_file)
        hts = rio_open(htinfo[0], band=htinfo[1])
    
    elif dem_type == 'csv':
        # Heights are in the .csv file
        hts = pd.read_csv(dem_file)['Hgt_m'].values

    elif dem_type == 'interpolate':
        # heights will be vertically interpolated to the heightlvs
        hts = None

    elif (dem_type == 'download') or (dem_type == 'dem'):
        if ~os.path.exists(dem_file):
            download_dem(
                ll_bounds, 
                writeDEM = True,
                outName=dem_file,
            )
        #TODO: interpolate heights to query lats/lons
        # Interpolate to the query points
        hts = interpolateDEM(
            dem_file,
            lats, 
            lons,
        )

    else:
        raise RuntimeError('dem_type is not valid')

    return hts


def download_dem(
    ll_bounds,
    save_flag='new',
    writeDEM=False,
    outName='warpedDEM',
    buf=0.02
):
    '''  Download a DEM if one is not already present. '''
    folder = os.path.dirname(outName)
    # inExtent is SNWE
    # dem-stitcher wants WSEN
    bounds = [
        np.floor(ll_bounds[2]) - buf, np.floor(ll_bounds[0]) - buf,
        np.ceil(ll_bounds[3]) + buf, np.ceil(ll_bounds[1]) + buf
    ]

    zvals, metadata = stitch_dem(
        bounds,
        dem_name='glo_30',
        dst_ellipsoidal_height=True,
        dst_area_or_point='Area',
    )
    dem_file = os.path.join(folder, 'GLO30_fullres_dem.tif')
    if writeDEM:
        with rasterio.open(dem_file, 'w', **metadata) as ds:
            ds.write(zvals, 1)
            ds.update_tags(AREA_OR_POINT='Point')

    return 

