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
    Function to return heights from a Digital Elevation Model (DEM), either from an existing file
    or by downloading one if needed.

    Parameters:
        - ll_bounds (tuple): Lower left and upper right bounds of the area of interest, in the format (lonmin, latmin, lonmax, latmax).
        - dem_type (str): Type of DEM to retrieve. Options are: 'hgt' (heights already stored in a .hgt file), 'csv' (heights stored in a .csv file), 'interpolate' (heights to be vertically interpolated), 'download' or 'dem' (download a DEM and interpolate to query points).
        - dem_file (str): File name or path for the DEM file to read or download.
        - lats (array or None): Array of latitudes for query points if dem_type is 'download' or 'dem'. Defaults to None.
        - lons (array or None): Array of longitudes for query points if dem_type is 'download' or 'dem'. Defaults to None.

    Returns:
        - hts (array or None): Array of heights from the DEM, or None if dem_type is 'interpolate'.
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
    '''
    Download a Digital Elevation Model (DEM) if it is not already present.

    Parameters:
        - ll_bounds (tuple): Lower left and upper right bounds of the area of interest, in the format (lonmin, latmin, lonmax, latmax).
        - writeDEM (bool): Whether to write the downloaded DEM to a file. Defaults to False.
        - outName (str): File name or path for the downloaded DEM file. Defaults to 'warpedDEM'.
        - buf (float): Buffer size to add to the bounds of the area of interest. Defaults to 0.02.
        - overwrite (bool): Whether to overwrite an existing DEM file with the same name as outName. Defaults to False.

    Returns:
        - zvals (array): 2D array of elevation values from the DEM.
        - metadata (dict): Dictionary of metadata for the DEM file.
    '''
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
