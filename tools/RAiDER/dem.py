#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
import re
import requests
import time

import multiprocessing as mp
import numpy as np
import pandas as pd

from osgeo import gdal
from pyproj import Proj
from shapely.geometry import shape, Polygon

import RAiDER.utilFcns

from RAiDER.interpolator import interpolateDEM
from RAiDER.logger import *
from RAiDER.utilFcns import gdal_open, gdal_extents


_DEM = "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1_E&west={}&south={}&east={}&north={}&outputFormat=GTiff"
_maxDEMSize = 2250000

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
        checkDEM=True,
        outName=os.path.join(os.getcwd(), 'warpedDEM'),
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
                # Use the existing DEM!
                _, _, _, geoProj, trans, noDataVal, _ = readRaster(outName)
                out = gdal_open(outName)
                logger.info('I am using an existing DEM')

        except AttributeError:
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
        fname = os.path.split(outName)[-1]
        full_res_dem = getDEM(inExtent, folder)
        _, _, _, geoProj, trans, noDataVal, _ = readRaster(full_res_dem)
        out = gdal_open(full_res_dem)
        logger.info('I am downloading a new DEM')

    out = out[::-1]
    # Interpolate to the query points
    logger.debug('Beginning interpolation')
    outInterp = interpolateDEM(out, np.stack((lats, lons), axis=-1), inExtent)
    logger.debug('Interpolation finished')

    # Write the DEM to requested location
    if save_flag == 'new':
        logger.debug('Saving DEM to disk')
        # ensure folders are created
        os.makedirs(folder, exist_ok=True)

        # Need to ensure that noData values are consistently handled and
        # can be passed on to GDAL
        if outInterp.ndim == 2:
            RAiDER.utilFcns.writeArrayToRaster(outInterp, outName, noDataValue=noDataVal)
        elif outInterp.ndim == 1:
            RAiDER.utilFcns.writeArrayToFile(lons, lats, outInterp, outName, noDataValue=noDataVal)
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
        if isinstance(lats, tuple) and len(lats) == 2:
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


def getDEM(extent, out_dir = os.getcwd(), num_threads = None):
    ''' 
    Get a DEM, chunking if needed 
    
    Parameters
    __________
    extent      - A list containing [lat_min, lat_max, lon_min, lon_max]
    out_dir     - Directory to write the full-resolution DEM
    num_threads - Number of threads to use when warping

    Returns
    _______
    final_dem_name - The name of the downloaded file

    '''
    if num_threads is None:
        num_threads = mp.cpu_count()*3//4

    lat_min, lat_max, lon_min, lon_max = extent
    query_area = getArea(extent)
    if query_area > _maxDEMSize:
        logger.warning(
            'Query area encompasses {} km^2, supersedes DEM maximum download'
            'area of 225000km, so I will download the DEM in chunks'.format(query_area)
        )        

    Nchunks = max(int(np.ceil(query_area/225000)) + 1, 2)
    chunk_size = (lon_max - lon_min)/Nchunks
    lon_starts = np.arange(lon_min, lon_max, chunk_size)

    # Download the DEM (in chunks if necessary)
    final_dem_name = os.path.join(out_dir, 'SRTM_1Sec.dem')
    chunked_files = []
    for i, L in enumerate(lon_starts):
        chunk_extent = [L, lat_min, L + chunk_size, lat_max]

        # Do not create temp file if chunking not necessary
        if len(lon_starts) > 2:
            dem_raster = 'tempdem_p{}_SRTM_1Sec.dem'.format(i)
        else:
            dem_raster = final_dem_name

        filename = os.path.join(out_dir, dem_raster)
        chunked_files.append(filename)

        dload_dem(chunk_extent, filename = dem_raster)

    # Tile chunked products together after last iteration (if necessary)
    if i > 1:
        gdal.Warp(
            final_dem_name, 
            chunked_files
#            options = gdal.WarpOptions(
#                 multithread=True, 
#                 options=['NUM_THREADS={}'.format(num_threads)]
#             )
        )

        # remove temp files
        [os.remove(i) for i in chunked_files]

    return final_dem_name


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
            (lat_max+lat_min)/2,
            (lon_max+lon_min)/2
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
    shape_area = shape(cop).area/1e6  # area in km^2

    return shape_area


def dload_dem(extent, filename = None):
    r = requests.get(_DEM.format(*extent), allow_redirects=True)
    open(filename, 'wb').write(r.content)
    del r


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


def readRaster(filename, band_num=None):
    '''
    Read a GDAL VRT file and return its attributes
    '''
    try:
        ds = gdal.Open(filename, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError('readRaster: cannot find file {}'.format(filename))
    except Exception as e:
        ds = None
        raise RuntimeError('readRaster: cannot open file {}. Reason: {}'.format(filename, e))

    xSize = ds.RasterXSize
    ySize = ds.RasterYSize
    geoProj = ds.GetProjection()
    trans = ds.GetGeoTransform()
    Nbands = ds.RasterCount

    # Read a band if I can
    if band_num is None:
        band_num = 1
        print('Using band one for dataType')
    try:
        dType = ds.GetRasterBand(band_num).DataType
        noDataVal = ds.GetRasterBand(band_num).GetNoDataValue()
        print('Could not access band {}, skipping noDataValue and dType'.format(band_num))
    except AttributeError:
        dType = None
        noDataVal = None

    ds = None

    return xSize, ySize, dType, geoProj, trans, noDataVal, Nbands
