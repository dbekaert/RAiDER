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


# Stamped into DEMs this module writes, so a later run can tell what datum a
# cached file is in. Files written before this tag existed carry ellipsoidal
# heights (dst_ellipsoidal_height was True), which is *not* what callers now
# assume, hence the warning in _check_cached_dem_datum().
DEM_DATUM_TAG = 'RAIDER_HEIGHT_DATUM'
DEM_DATUM = 'geoid'


def _check_cached_dem_datum(dem_path: Path) -> None:
    """Warn if a reused DEM may predate the switch to geoid-referenced heights.

    download_dem() reuses whatever is already at dem_path without re-reading
    its provenance, and a DEM written by an earlier RAiDER holds *ellipsoidal*
    heights. Treating those as geoid displaces them by the geoid undulation --
    ~35 m in CONUS -- and for StationFile that wrong datum then gets written
    into the user's CSV as an authoritative Hgt_datum label.
    """
    try:
        with rasterio.open(dem_path) as ds:
            datum = ds.tags().get(DEM_DATUM_TAG)
    except Exception:  # unreadable tags shouldn't break the read that follows
        return

    if datum == DEM_DATUM:
        return

    logger.warning(
        'Reusing DEM %s, which carries no RAiDER height-datum tag. DEMs '
        'written by earlier versions hold ellipsoidal heights, but heights '
        'are now read as geoid-referenced -- a ~35 m difference in CONUS. '
        'Delete it to re-download if it predates this version.',
        dem_path,
    )


def download_dem(
    ll_bounds: Union[tuple, List, np.ndarray] = None,
    dem_path: Path = Path('warpedDEM.rdr'),
    overwrite: bool = False,
    writeDEM: bool = False,
    buf: float = 0.02,
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
        _check_cached_dem_datum(dem_path)
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

        # GLO-30 is distributed against the EGM2008 geoid, which is the datum
        # RAiDER reads geoid heights in, and the datum the weather-model cube
        # is sampled against. Taking it as-is means the common path converts
        # nothing at all; asking dem-stitcher for ellipsoidal instead would
        # fetch a geoid grid to do a conversion that sampling then has to
        # undo, spending a download and two interpolations to arrive back
        # where it started.
        zvals, metadata = stitch_dem(
            list(bounds),
            dem_name='glo_30',
            dst_ellipsoidal_height=False,
            dst_area_or_point='Area',
        )
        metadata = cast(RIO.Profile, metadata)
        if writeDEM:
            with rasterio.open(dem_path, 'w', **metadata) as ds:
                ds.write(zvals, 1)
                ds.update_tags(AREA_OR_POINT='Point', **{DEM_DATUM_TAG: DEM_DATUM})
            logger.info('Wrote DEM: %s', dem_path)

    return zvals, metadata
