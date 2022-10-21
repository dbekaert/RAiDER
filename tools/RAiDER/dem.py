#!/usr/bin/env python3
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
from pyproj import Proj
from shapely.geometry import shape, Polygon
from dem_stitcher.stitcher import stitch_dem

import RAiDER.utilFcns

from RAiDER.interpolator import interpolateDEM
from RAiDER.logger import logger
from RAiDER.utilFcns import rio_open, rio_profile, rio_extents, get_file_and_band


def getHeights(lats, lons, heights, useWeatherNodes=False):
    '''
    Fcn to return heights from a DEM, either one that already exists
    or will download one if needed.
    '''
    height_type, height_data = heights
    if isinstance(lats, str):
        latinfo = get_file_and_band(lats)
        lats = rio_open(latinfo[0], band=latinfo[1])

    if isinstance(lons, str):
        loninfo = get_file_and_band(lons)
        lons = rio_open(loninfo[0], band=loninfo[1])

    in_shape = lats.shape

    if height_type == 'dem':
        try:
            htinfo = get_file_and_band(height_data)
            hts = rio_open(htinfo[0], band=htinfo[1])
            assert hts.shape == lats.shape
        except BaseException:
            logger.warning(
                'File %s could not be opened; requires GDAL-readable file.',
                height_data, exc_info=True
            )
            logger.info('Proceeding with DEM download')
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
        for f in height_data:
            data = pd.read_csv(f)
            lats = data['Lat'].values
            lons = data['Lon'].values
            hts = download_dem(lats, lons, outName=f, save_flag='merge')
            data['Hgt_m'] = hts
            data.to_csv(f)
    elif height_type == 'pandas':
        data = pd.read_csv(height_data[0])
        hts = data['Hgt_m'].values
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
    writeDEM=False,
    outName='warpedDEM',
    buf=0.02
):
    '''  Download a DEM if one is not already present. '''

    # Get the lat/lon extents of the query points
    inExtent = getBufferedExtent(lats, lons, buf=buf)

    # Check if the DEM exists, use it if I can, otherwise download a new one
    if os.path.exists(outName):
        do_download = False
        logger.warning(
            'A DEM already exists in {}, checking extents'
            .format(os.path.dirname(outName))
        )

        try:
            bounds = rio_extents(rio_profile(outName))
            lons = bounds[:2]
            lats = bounds[2:]
            if isOutside(
                inExtent,
                getBufferedExtent(
                    lats,
                    lons,
                    buf=buf
                )
            ):
                raise ValueError(
                    'Existing DEM does not cover the area of the input lat/lon '
                    'points; either move the DEM, delete it, or change the input '
                    'points.'
                )
            else:
                # Use the existing DEM!
                ds = rasterio.open(outName)
                noDataVal = ds.nodatavals[0]
                ds.close()

                out = rio_open(outName)
                save_flag = False
                logger.info('I am using an existing DEM')

        except AttributeError:
            out = rio_open(outName)
            if lats.shape==out.shape:
                do_download = False
                save_flag = False
            else:
                logger.warning(
                    'Existing DEM does not contain geo-referencing info, so '
                    'I will download a new one.'
                )
                do_download = True
        except OSError:
            try:
                hgts = RAiDER.utilFcns.read_hgt_file(outName)
                return hgts
            except KeyError:
                logger.warning('The station file does not contain height information, I will download it')
                do_download = True
    else:
        do_download = True

    # Otherwise download a new DEM
    if do_download:
        folder = os.sep.join(os.path.split(outName)[:-1])
        # inExtent is SNWE
        # dem-stitcher wants WSEN
        bounds = [np.floor(inExtent[2]), np.floor(inExtent[0]),
                 np.ceil(inExtent[3]), np.ceil(inExtent[1]),]

        zvals, metadata = stitch_dem(bounds,
                            dem_name='glo_30',
                            dst_ellipsoidal_height=True,
                            dst_area_or_point='Area',
                            )
        if writeDEM:
            with rasterio.open('GLO30_fullres_dem.tif', 'w', **metadata) as ds:
                ds.write(zvals, 1)

    # Interpolate to the query points
    logger.debug('Beginning interpolation')
    outInterp = interpolateDEM(
        zvals,
        np.stack((lats, lons), axis=-1),
        inExtent,
        method='linear',
    )
    logger.debug('Interpolation finished')

    # Write the DEM to requested location
    if save_flag == 'new':
        logger.debug('Saving DEM to disk')
        # ensure folders are created
        os.makedirs(folder, exist_ok=True)

        # Need to ensure that noData values are consistently handled and
        # can be passed on to GDAL
        if outInterp.ndim == 2:
            metadata['height'] = outInterp.shape[0]
            metadata['width'] = outInterp.shape[1]
            metadata['transform'] = None
            with rasterio.open(outName, 'w', **metadata) as ds:
                ds.write(outInterp, 1)
        elif outInterp.ndim == 1:
            RAiDER.utilFcns.writeArrayToFile(
                lons,
                lats,
                outInterp,
                outName,
                noDataValue=noDataVal
            )
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


def getBufferedExtent(lats, lons=None, buf=0.):
    '''
    get the bounding box around a set of lats/lons
    '''
    if lons is None:
        lats, lons = lats[..., 0], lons[..., 1]

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
        if (isinstance(lats, tuple) or isinstance(lats,list)) and len(lats) == 2:
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


def getArea(extent):
    '''
    Get the area in square km encompassed by a lat/lon bounding box
    '''
    lat_min, lat_max, lon_min, lon_max = extent

    # use equal area projection centered on/bracketing AOI
    pa = Proj(
        "+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(
            lat_min,
            lat_max,
            (lat_max + lat_min) / 2,
            (lon_max + lon_min) / 2
        )
    )

    # Use shapely to get coordinates along box
    bbox = Polygon(
        np.column_stack(
            (
                np.array(
                    [
                        lon_min,
                        lon_max,
                        lon_max,
                        lon_min,
                        lon_min
                    ]
                ),
                np.array(
                    [
                        lat_min,
                        lat_min,
                        lat_max,
                        lat_max,
                        lat_min
                    ]
                )
            )
        )
    )

    lon, lat = bbox.exterior.coords.xy
    x, y = pa(lon, lat)
    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
    shape_area = shape(cop).area / 1e6  # area in km^2

    return shape_area
