#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os

import pandas as pd

from RAiDER.utilFcns import rio_stats, get_file_and_band


def bounds_from_latlon_rasters(latfile, lonfile):
    '''
    Parse lat/lon/height inputs and return
    the appropriate outputs
    '''
    latinfo = get_file_and_band(latfile)
    loninfo = get_file_and_band(lonfile)
    lat_stats, lat_proj, _ = rio_stats(latinfo[0], band=latinfo[1])
    lon_stats, lon_proj, _ = rio_stats(loninfo[0], band=loninfo[1])

    if lat_proj != lon_proj:
        raise ValueError('Projection information for Latitude and Longitude files does not match')

    # TODO - handle dateline crossing here
    snwe = (lat_stats.min, lat_stats.max,
            lon_stats.min, lon_stats.max)

    fname = os.path.basename(latfile).split('.')[0]
    
    return lat_proj, snwe, fname


def bounds_from_csv(station_file):
    '''
    station_file should be a comma-delimited file with at least "Lat" 
    and "Lon" columns, which should be EPSG: 4326 projection (i.e WGS84)
    '''
    stats = pd.read_csv(fname).drop_duplicates(subset=["Lat", "Lon"])
    if 'Hgt_m' in stats.columns:
        use_csv_heights = True
    snwe = [stats['Lat'].min(), stats['Lat'].max(), stats['Lon'].min(), stats['Lon'].max()]
    fname = os.path.basename(station_file).split('.')[0]
    return snwe, fname, use_csv_heights
