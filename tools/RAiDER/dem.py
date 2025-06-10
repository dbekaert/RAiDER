# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from pathlib import Path
from typing import List, Optional, Union, cast

import numpy as np
import rasterio
from dem_stitcher.stitcher import stitch_dem

from RAiDER.logger import logger
from RAiDER.types import BB, RIO
from RAiDER.utilFcns import rio_open


def download_dem(
        ll_bounds: Union[tuple, List, np.ndarray]=None,
        dem_path: Path=Path('warpedDEM.dem'),
        overwrite: bool=False,
        writeDEM: bool=False,
        buf: float=0.02,
    ) -> tuple[np.ndarray, Optional[RIO.Profile]]:
    """Download a DEM if one is not already present.

    Args:
        ll_bounds: list/ndarry of floats    - lat/lon bounds of the area to download. Values should be ordered in the following way: [S, N, W, E]
        dem_path: string                    - Path to write DEM file
        overwrite: bool                     - overwrite existing DEM
        writeDEM: bool                      - write the DEM to file
        buf: float                          - buffer to add to the bounds
    Returns:
        zvals: np.array                 - DEM heights
        metadata:                       - metadata for the DEM
    """
    if dem_path.exists():
        download = overwrite
    else:
        download = True

    if download and ll_bounds is None:
        raise ValueError('download_dem: Either an existing file or lat/lon bounds must be passed')

    if not download:
        logger.info('Using existing DEM: %s', dem_path)
        zvals, metadata = rio_open(dem_path)
    else:
        # download the dem
        # inExtent is SNWE
        # dem-stitcher wants WSEN
        bounds: BB.WSEN = (
            np.floor(ll_bounds[2]) - buf,
            np.floor(ll_bounds[0]) - buf,
            np.ceil(ll_bounds[3]) + buf,
            np.ceil(ll_bounds[1]) + buf,
        )

        zvals, metadata = stitch_dem(
            list(bounds),
            dem_name='glo_30',
            dst_ellipsoidal_height=True,
            dst_area_or_point='Area',
        )
        metadata = cast(RIO.Profile, metadata)
        if writeDEM:
            with rasterio.open(dem_path, 'w', **metadata) as ds:
                ds.write(zvals, 1)
                ds.update_tags(AREA_OR_POINT='Point')
            logger.info('Wrote DEM: %s', dem_path)

    return zvals, metadata
