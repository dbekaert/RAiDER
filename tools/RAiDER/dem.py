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
        zvals, metadata = download_dem(ll_bounds, writeDEM=True, outName=dem_file)

        #TODO: check this
        lons, lats = np.meshgrid(lons, lats)
        # Interpolate to the query points
        hts = interpolateDEM(zvals, metadata['transform'], (lats, lons), method='nearest')

    return hts


def download_dem(
    ll_bounds,
    writeDEM=False,
    outName='warpedDEM',
    buf=0.02,
    overwrite=False,
):
    """  Download a DEM if one is not already present. """
    if os.path.exists(outName) and not overwrite:
        logger.info('Using existing DEM: %s', outName)
        zvals, metadata = rio_open(outName, returnProj=True)

    else:
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
        if writeDEM:
            with rasterio.open(outName, 'w', **metadata) as ds:
                ds.write(zvals, 1)
                ds.update_tags(AREA_OR_POINT='Point')
            logger.info('Wrote DEM: %s', outName)

    return zvals, metadata
