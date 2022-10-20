#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import numpy as np
import pandas as pd

from RAiDER.utilFcns import gdal_open, gdal_stats, get_file_and_band


def readLL(*args):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    if len(args[0]) == 2:
        # If they are files, open them for stats
        flag = 'files'

        latinfo= get_file_and_band(args[0][0])
        loninfo = get_file_and_band(args[0][1])
        lat_stats = gdal_stats(latinfo[0], band=latinfo[1])
        lon_stats = gdal_stats(loninfo[0], band=loninfo[1])
        snwe = (lat_stats[0][0], lat_stats[0][1],
                lon_stats[0][0], lon_stats[0][1])

        lats, lons, llproj = readLLFromBBox(snwe)
        fname = os.path.basename(args[0][0]).split('.')[0]

    elif len(args[0]) == 4:
        flag = 'bounding_box'
        lats, lons, llproj = readLLFromBBox(*args)
        fname = '_'.join([str(int(a)) for a in [a for a in args][0]])

    elif isinstance(args[0], str):
        flag = 'station_file'
        lats, lons, llproj = readLLFromStationFile(*args)
        fname = os.path.basename(args[0]).split('.')[0]

    else:
        raise RuntimeError('llreader: Cannot parse query region: {}'.format(args))

    bounds = (np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons))
    # If files, pass file names instead of arrays 
    if flag == "files":
        lats = args[0][0]
        lons = args[0][1]


    pnts_file_name = 'query_points_' + fname + '.h5'

    return lats, lons, llproj, bounds, flag, pnts_file_name


def readLLFromLLFiles(latfile, lonfile):
    ''' Read ISCE-style files having pixel lat and lon in radar coordinates '''
    lats, llproj, _ = gdal_open(latfile, returnProj=True)
    lons, llproj2, _ = gdal_open(lonfile, returnProj=True)
    lats[lats == 0.] = np.nan
    lons[lons == 0.] = np.nan
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
    stats = pd.read_csv(fname).drop_duplicates(subset=["Lat", "Lon"])
    return stats['Lat'].values, stats['Lon'].values, 'EPSG:4326'


def forceNDArray(arg):
    if arg is None:
        return None
    else:
        return np.array(arg)
