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


def checkArgs(args, p):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''

    # Argument checking
    if args.heightlvs is not None and args.outformat != 'hdf5':
       raise ValueError('HDF5 must be used with height levels')
    if args.area is None and args.bounding_box is None and args.wmnetcdf is None and args.station_file is None:
       raise ValueError('You must specify one of the following: \n \
             (1) lat/lon files, (2) bounding box, (3) weather model files, or\n \
             (4) station file containing Lat and Lon columns')
    if args.model not in AllowedModels():
       raise NotImplementedError('Model {} is not an implemented model type'.format(args.model))
    if args.model == 'WRF' and args.wrfmodelfiles is None:
       p.error('Argument --wrfmodelfiles required with --model WRF')

    # flag depending on the type of input
    if args.area is not None:
        flag = 'files'
    elif args.bounding_box is not None:
        flag = 'bounding_box'
    elif args.station_file is not None:
        flag = 'station_file'
    else: 
        flag = None

    # other
    out = args.out
    if out is None:
        out = os.getcwd()
    download_only = args.download_only
    verbose = args.verbose

    # Line of sight calc
    if args.lineofsight is not None:
        los = ('los', args.lineofsight)
    elif args.statevectors is not None:
        los = ('sv', args.statevectors)
    else:
        from RAiDER.constants import Zenith
        los = Zenith

    # Area
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

    # Weather
    weather_model_name = args.model.upper().replace('-','')
    model_module_name, model_obj = RAiDER.utilFcns.modelName2Module(args.model)
    if args.model == 'WRF':
       weathers = {'type': 'wrf', 'files': args.wrfmodelfiles,
                   'name': 'wrf'}
    elif args.model=='pickle':
        weathers = {'type':'pickle', 'files': args.pickleFile, 'name': 'pickle'}
    elif args.wmnetcdf is not None:
        weathers = {'type': model_obj(), 'files': args.wmnetcdf,
                    'name': args.model}
    else:
        weathers = {'type': model_obj(), 'files': None,
                    'name': args.model}

    # output file format
    if args.outformat is None:
       if args.heightlvs is not None: 
          outformat = 'hdf5'
       elif args.station_file is not None: 
          outformat = 'netcdf'
       else:
          outformat = 'ENVI'
    else:
       outformat = args.outformat
    # ensuring consistent file extensions
    #outformat = output_format(outformat)

    # zref
    zref = args.zref

    # handle the datetimes requested
    datetimeList = [d + args.time for d in args.dateList]

    # output filenames
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

    if args.wmLoc is not None:
       wmLoc = args.wmLoc
    else:
       wmLoc = os.path.join(args.out, 'weather_files')

    if args.heightlvs is not None: 
       if outformat.lower() != 'hdf5':
          print("WARNING: input arguments require HDF5 output file type; changing outformat to HDF5")
       outformat = 'hdf5'
    elif args.station_file is not None:
       if outformat.lower() != 'netcdf':
          print("WARNING: input arguments require HDF5 output file type; changing outformat to HDF5")
       outformat = 'netcdf'
    else:
       if outformat.lower() == 'hdf5':
          print("WARNING: output require raster output file; changing outformat to ENVI")
          outformat = 'ENVI'
       else:
          outformat = outformat.lower()

    # parallelization
    parallel = True if not args.no_parallel else False

    return los, lat, lon, heights, flag, weathers, wmLoc, zref, outformat, datetimeList, out, download_only, parallel, verbose, wetNames, hydroNames


def output_format(outformat):
    """
        Reduce the outformat strings users can specifiy to a select consistent set that can be used for filename extensions.
    """
    # convert the outformat to lower letters
    outformat = outformat.lower()

    # capture few specific cases:
    outformat_dict = {}
    outformat_dict['hdf5'] = 'h5'
    outformat_dict['hdf'] = 'h5'
    outformat_dict['h5'] = 'h5'
    outformat_dict['envi'] = 'envi'

    try:
        outformat = outformat_dict[outformat]
    except:
        raise NotImplementedError
    return outformat
