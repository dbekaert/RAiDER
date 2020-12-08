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
import pandas as pd

from RAiDER.logger import *
from RAiDER.ioFcns import gdal_open


def readLL(*args):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    if len(args[0]) == 2:
        # If they are files, open them
        flag = 'files'
        lats, lons, llproj = readLLFromLLFiles(*args[0])
    elif len(args[0]) == 4:
        flag = 'bounding_box'
        lats, lons, llproj = readLLFromBBox(*args)
    elif isinstance(args[0], str):
        flag = 'station_file'
        lats, lons, llproj = readLLFromStationFile(*args)
    else:
        raise RuntimeError('llreader: Cannot parse query region: {}'.format(args))

    lats, lons = forceNDArray(lats), forceNDArray(lons)
    bounds = (np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons))

    return lats, lons, llproj, bounds, flag


def readLLFromLLFiles(latfile, lonfile):
    ''' Read ISCE-style files having pixel lat and lon in radar coordinates '''
    lats, llproj, _ = gdal_open(latfile, returnProj=True)
    lons, llproj2, _ = gdal_open(lonfile, returnProj=True)
    if llproj != llproj2:
        raise ValueError('The projection of the lat and lon files are not compatible')
    return lats, lons, llproj


def readLLFromBBox(bbox):
    ''' Convert string bounding box to numpy lat/lon arrays '''
    S, N, W, E = bbox
    lats = np.array([float(S), float(N)])
    lons = np.array([float(W), float(E)])
    return lats, lons, 'EPSG:4326'


def readLLFromStationFile(fname):
    '''
    Helper fcn for checking argument compatibility
    '''
    stats = pd.read_csv(fname)
    return stats['Lat'].values, stats['Lon'].values, 'EPSG:4326'


def forceNDArray(arg):
    if arg is None:
        return None
    else:
        return np.array(arg)
