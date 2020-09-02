#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import os
from datetime import datetime

import numpy as np

import RAiDER.utilFcns
from RAiDER.llreader import readLL


def checkArgs(args, p):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''

    # Argument checking
    if args.heightlvs is not None:
        if args.outformat is not None:
            if args.outformat.lower() != 'hdf5':
                raise RuntimeError('HDF5 must be used with height levels')

    # Area
    # flag depending on the type of input
    if args.latlon is not None:
        flag = 'files'
    elif args.bbox is not None:
        flag = 'bounding_box'
    elif args.station_file is not None:
        flag = 'station_file'
    else:
        flag = None

    if args.latlon is not None:
        lat, lon, latproj, lonproj, bounds = readLL(*args.latlon)
    elif args.bbox is not None:
        lat, lon, latproj, lonproj, bounds = readLL(*args.bbox)
    elif args.station_file is not None:
        lat, lon, latproj, lonproj, bounds = readLL(args.station_file)
    elif args.files is None:
        raise NotImplementedError("Reading lat/lon data from files is not implemented")
    else:
        raise RuntimeError('You must specify an area of interest')

    if (np.min(lat) < -90) | (np.max(lat) > 90):
        raise RuntimeError('Lats are out of N/S bounds; are your lat/lon coordinates switched?')

    # Line of sight calc
    los = args.lineofsight

    # Weather
    weather_model_name = args.model
    if weather_model_name == 'WRF' and args.files is None:
        raise RuntimeError('Argument --files is required with --model WRF')
    _, model_obj = RAiDER.utilFcns.modelName2Module(args.model)
    if args.model == 'WRF':
        weathers = {'type': 'wrf', 'files': args.files,
                    'name': 'wrf'}
    elif args.model == 'HDF5':
        weathers = {'type': 'HDF5', 'files': args.files,
                    'name': args.model}
    else:
        try:
            weathers = {'type': model_obj(), 'files': args.files,
                        'name': args.model}
        except:
            raise NotImplementedError('{} is not implemented'.format(weather_model_name))

    # zref
    zref = args.zref

    # handle the datetimes requested
    datetimeList = [datetime.combine(d, args.time) for d in args.dateList]

    # Misc
    download_only = args.download_only
    verbose = args.verbose
    useWeatherNodes = flag == 'bounding_box'

    # Output
    out = args.out
    if out is None:
        out = os.getcwd()
    if args.outformat is None:
        if args.heightlvs is not None:
            outformat = 'hdf5'
        elif args.station_file is not None:
            outformat = 'csv'
        elif useWeatherNodes:
            outformat = 'hdf5'
        else:
            outformat = 'envi'
    else:
        outformat = args.outformat.lower()
    if args.wmLoc is not None:
        wmLoc = args.wmLoc
    else:
        wmLoc = os.path.join(args.out, 'weather_files')

    if not os.path.exists(wmLoc):
        os.mkdir(wmLoc)

    wetNames, hydroNames = [], []
    for time in datetimeList:
        if flag == 'station_file':
            wetFilename = os.path.join(out, '{}_Delay_{}_Zmax{}.csv'
                                       .format(weather_model_name, time.strftime('%Y%m%dT%H%M%S'), zref))
            hydroFilename = wetFilename

            # copy the input file to the output location for editing
            import pandas as pd
            indf = pd.read_csv(args.station_file)
            indf.to_csv(wetFilename, index=False)
        else:
            wetFilename, hydroFilename = \
                RAiDER.utilFcns.makeDelayFileNames(time, los.getLOSType(), outformat, weather_model_name, out)

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    # DEM
    if args.dem is not None:
        heights = ('dem', args.dem)
    elif args.heightlvs is not None:
        heights = ('lvs', args.heightlvs)
    elif flag == 'station_file':
        heights = ('merge', wetNames)
    elif useWeatherNodes:
        heights = ('skip', None)
    else:
        heights = ('download', os.path.join(out, 'geom', 'warpedDEM.dem'))

    return los, lat, lon, bounds, heights, flag, weathers, wmLoc, zref, outformat, datetimeList, out, download_only, verbose, wetNames, hydroNames
