#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
import os

from RAiDER.llreader import readLL
from RAiDER.models.allowed import AllowedModels
import RAiDER.utilFcns
from RAiDER.constants import Zenith

def checkArgs(args, p):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''

    # Argument checking
    if args.heightlvs is not None:
        if args.outformat is not None:
            if args.outformat.lower() != 'hdf5':
                print('HDF5 must be used with height levels')
                args.outformat= 'hdf5'
    if args.model not in AllowedModels():
       raise NotImplementedError('Model {} has not been implemented'.format(args.model))
    if args.model == 'WRF' and args.files is None:
       p.error('Argument --files is required with --model WRF')

    ## Area
    # flag depending on the type of input
    if args.area is not None:
        flag = 'files'
    elif args.bounding_box is not None:
        flag = 'bounding_box'
    elif args.station_file is not None:
        flag = 'station_file'
    else: 
        flag = None

    if args.area is not None:
        lat, lon, latproj, lonproj = readLL(*args.area)
    elif args.bounding_box is not None:
        lat, lon, latproj, lonproj = readLL(*args.bounding_box)
    elif args.station_file is not None:
        lat, lon, latproj, lonproj = RAiDER.llreader.readLL(args.station_file)
    else:
        lat = lon = None
    from numpy import min, max
    if (min(lat) < -90) | (max(lat)>90):
        raise RuntimeError('Lats are out of N/S bounds; are your lat/lon coordinates switched?')

    # Line of sight calc
    if args.lineofsight is not None:
        los = ('los', args.lineofsight)
    elif args.statevectors is not None:
        los = ('sv', args.statevectors)
    else:
        los = Zenith

    # Weather
    weather_model_name = args.model.upper().replace('-','')
    model_module_name, model_obj = RAiDER.utilFcns.modelName2Module(args.model)
    if args.model == 'WRF':
       weathers = {'type': 'wrf', 'files': args.files,
                   'name': 'wrf'}
    elif args.model=='HDF5':
            weathers = {'type': 'HDF5', 'files': args.files,
                    'name': args.model}
    else:
        try:
            weathers = {'type': model_obj(), 'files': args.files,
                    'name': args.model}
        except:
            raise NotImplemented('{} is not implemented'.format(weather_model_name))

    # zref
    zref = args.zref

    # handle the datetimes requested
    datetimeList = [d + args.time for d in args.dateList]

    ## Misc
    download_only = args.download_only
    verbose = args.verbose

    ## Output
    out = args.out
    if out is None:
        out = os.getcwd()
    if args.outformat is None:
       if args.heightlvs is not None: 
          outformat = 'hdf5'
       elif args.station_file is not None: 
          outformat = 'csv'
       elif los is Zenith:
          outformat='hdf5'
       else:
          outformat = 'envi'
    else:
       outformat = args.outformat.lower()
    if args.wmLoc is not None:
       wmLoc = args.wmLoc
    else:
       wmLoc = os.path.join(args.out, 'weather_files')


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
                RAiDER.utilFcns.makeDelayFileNames(time, los, outformat, weather_model_name, out)

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    # DEM
    if args.dem is not None:
        heights = ('dem', args.dem)
    elif args.heightlvs is not None:
        heights = ('lvs', args.heightlvs)
    elif flag=='station_file':
        heights = ('merge', wetNames)
    else:
        heights = ('download', 'geom/warpedDEM.dem')

    return los, lat, lon, heights, flag, weathers, wmLoc, zref, outformat, datetimeList, out, download_only, verbose, wetNames, hydroNames

