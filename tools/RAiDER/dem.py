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

import numpy as np
from osgeo import gdal

import RAiDER.utilFcns

from RAiDER.interpolator import interpolateDEM
from RAiDER.logger import *
from RAiDER.utilFcns import gdal_open, gdal_extents


_DEM = "https://portal.opentopography.org/API/globaldem?demtype=SRTMGL1_E&west={}&south={}&east={}&north={}&outputFormat=GTiff"


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
        import pandas as pd
        for f in height_data:
            data = pd.read_csv(f)
            lats = data['Lat'].values
            lons = data['Lon'].values
            hts = download_dem(lats, lons, outName=f, save_flag='merge')
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
            hgts = RAiDER.utilFcns.read_hgt_file(outName)
            return hgts
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


def getDEM(extent, out_dir):
    """Download the DEM over bounding box."""
    # import dependencies
    from pyproj import Proj
    from shapely.geometry import shape
    from shapely.geometry import Polygon
    import glob

    # change order of extent to WESN
    extent = [extent[2], extent[3], extent[0], extent[1]]
    # Get area of bounding box
    # use equal area projection centered on/bracketing AOI
    pa = Proj("+proj=aea +lat_1={} +lat_2={} +lat_0={} +lon_0={}".format(min(extent[1::2]), max(extent[1::2]), 
        (max(extent[1::2])+min(extent[1::2]))/2, (max(extent[::2])+min(extent[::2]))/2))
    # Use shapely to get coordinates along box
    bbox = Polygon(np.column_stack((np.array([min(extent[::2]),max(extent[::2]),max(extent[::2]),min(extent[::2]),min(extent[::2])]), 
        np.array([min(extent[1::2]),min(extent[1::2]),max(extent[1::2]),max(extent[1::2]),min(extent[1::2])]))))
    lon, lat = bbox.exterior.coords.xy
    x, y = pa(lon, lat)
    cop = {"type": "Polygon", "coordinates": [zip(x, y)]}
    shape_area = shape(cop).area/1e6  # area in km^2

    chunking_size = 2
    # If area > 225000 km2, must split requests into chunks to successfully access data
    if shape_area > 225000:
        # Increase chunking size to discretize box into smaller grids
        logger.warning("User-defined bounds %dkm\u00b2 supersedes DEM maximum download area of 225000km\u00b2, must download in chunks", shape_area)
        chunking_size = int(np.ceil(shape_area/225000)) + 1

    # Determine number of iterations to download DEM
    bottomLeft = (min(extent[1::2]), min(extent[::2]))
    bottomRight = (min(extent[1::2]), max(extent[::2]))
    topLeft = (max(extent[1::2]), min(extent[::2]))
    cols = np.linspace(bottomLeft[1], bottomRight[1], num=chunking_size)
    rows = [bottomLeft[0], topLeft[0]]
    # Download in chunks (if necessary)
    chunked_files = []
    for i in enumerate (cols[:-1]):
        chunk_extent = [cols[i[0]], rows[0], cols[i[0]+1], rows[1]]
        r = requests.get(_DEM.format(*chunk_extent), allow_redirects=True)
        final_demname = get_filename_from_cd(r.headers.get('content-disposition'))
        dem_raster = get_filename_from_cd(r.headers.get('content-disposition'))
        # Do not create temp file if chunking not necessary
        if len(cols) > 2:
            dem_raster = 'tempdem_p{}'.format(i[0]) + dem_raster
        chunked_files.append(dem_raster)
        filename = os.path.join(out_dir, dem_raster)
        open(filename, 'wb').write(r.content)
        del r
    # Tile chunked products together after last iteration (if necessary)
    if len(cols) != 2:
        final_demname = os.path.join(out_dir, final_demname)
        gdal.Warp(final_demname, chunked_files)
        # remove temp files
        for i in glob.glob(os.path.join(out_dir, 'tempdem_p*')): os.remove(i)

    return filename


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
    if band_num is None:
        band_num = 1
        print('Using band one for dataType')

    dType = ds.GetRasterBand(band_num).DataType
    noDataVal = ds.GetRasterBand(band_num).GetNoDataValue()
    ds = None

    return xSize, ySize, dType, geoProj, trans, noDataVal, Nbands
