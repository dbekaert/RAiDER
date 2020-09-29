#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import logging
import os

import numpy as np

from RAiDER.utilFcns import gdal_open

log = logging.getLogger(__name__)


def readLL(*args):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    if len(args) == 2:
        flag = 'files'
    elif len(args) == 4:
        flag = 'bounding_box'
    elif len(args) == 1:
        flag = 'station_list'
    else:
        raise RuntimeError('llreader: Cannot parse args')

    # Lats/Lons
    if flag == 'files':
        # If they are files, open them
        lat, lon = args
        lats, llproj1, _ = gdal_open(lat, returnProj=True)
        lons, llproj2, _ = gdal_open(lon, returnProj=True)
        if llproj1 != llproj2:
            raise ValueError('The project of the lat and lon files are not compatible')
    elif flag == 'bounding_box':
        S, N, W, E = args
        lats = np.array([float(N), float(S)])
        lons = np.array([float(E), float(W)])
        llproj = 'EPSG:4326'
        if (lats[0] == lats[1]) | (lons[0] == lons[1]):
            raise RuntimeError('You have passed a zero-size bounding box: {}'
                               .format(args.bounding_box))
    elif flag == 'station_list':
        lats, lons = readLLFromStationFile(*args)
        llproj = 'EPSG:4326'
    else:
        # They'll get set later with weather
        lats = lons = None
        llproj = None
        #raise RuntimeError('readLL: unknown flag')

    [lats, lons] = enforceNumpyArray(lats, lons)
    bounds = (np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons))

    return lats, lons, latproj, lonproj, bounds


def readLLFromStationFile(fname):
    '''
    Helper fcn for checking argument compatibility
    '''
    try:
        import pandas as pd
        stats = pd.read_csv(fname)
        return stats['Lat'].values, stats['Lon'].values
    except:
        lats, lons = [], []
        with open(fname, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                lat, lon = [float(f) for f in line.split(',')[1:3]]
                lats.append(lat)
                lons.append(lon)
        return lats, lons
