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

from RAiDER.logger import logger
from RAiDER.utilFcns import rio_open


def download_dem(
        ll_bounds=None,
        writeDEM=False,
        outName='warpedDEM',
        buf=0.02,
        overwrite=False,
    ):
    """  
    Download a DEM if one is not already present. 
    Args:
            llbounds: list/ndarry of floats   -lat/lon bounds of the area to download. Values should be ordered in the following way: [S, N, W, E]
            writeDEM: boolean                 -write the DEM to file
            outName: string                   -name of the DEM file   
            buf: float                        -buffer to add to the bounds
            overwrite: boolean                -overwrite existing DEM
    Returns:
            zvals: np.array         -DEM heights
            metadata:               -metadata for the DEM
    """
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
