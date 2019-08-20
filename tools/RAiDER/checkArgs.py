#!/usr/bin/env python3
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import os
from RAiDER.util import gdal_trans
from RAiDER.llreader import readLL

def mangle_model_to_module(model_name):
    """Turn an arbitrary string into a module name.

    Takes as input a model name, which hopefully looks like ERA-I, and
    converts it to a module name, which will look like erai. I doesn't
    always produce a valid module name, but that's not the goal. The
    goal is just to handle common cases.
    """
    module_name = 'RAiDER.models.' + model_name.lower().replace('-', '')
    object_name = module_name + '.' + model_name.upper().replace('-', '')
    return module_name, object_name


def checkArgs(args, p):
    '''
    Helper fcn for checking argument compatibility and returns the 
    correct variables
    '''
    from datetime import datetime as dt

    if args.heightlvs is not None and args.outformat != 'hdf5':
       raise ValueError('HDF5 must be used with height levels')
    if args.area is None and args.bounding_box is None and args.wmnetcdf is None and args.station_file is None:
       raise ValueError('You must specify one of the following: \n \
             (1) lat/lon files, (2) bounding box, (3) weather model files, or\n \
             (4) station file containing Lat and Lon columns')

    # Line of sight
    if args.lineofsight is not None:
        los = ('los', args.lineofsight)
    elif args.statevectors is not None:
        los = ('sv', args.statevectors)
    else:
        from RAiDER.constants import Zenith
        los = Zenith

    # Area
    if args.area is not None:
        lat, lon = readLL(*args.area)
    elif args.bounding_box is not None:
        lat, lon = readLL(*args.bounding_box)
    elif args.station_file is not None:
        lat, lon = readLL(args.station_file)
    else:
        lat = lon = None

    if (lat[0] == lat[1]) | (lon[0]==lon[1]):
       raise RuntimeError('You have passed a zero-size bounding box: {}'.format(args.bounding_box))

    # write the lats/lons to a local file
    lonFileName = '{}_Lon_{}.dat'.format(args.model.lower().replace('-',''), 
                      dt.strftime(args.time, '%Y_%m_%d_T%H_%M_%S'))
    latFileName = '{}_Lat_{}.dat'.format(args.model.lower().replace('-',''), 
                      dt.strftime(args.time, '%Y_%m_%d_T%H_%M_%S'))

    #TODO
    #gdal_trans(lat, os.path.join(args.out, 'geom', latFileName), 'VRT')
    #gdal_trans(lon, os.path.join(args.out, 'geom', lonFileName), 'VRT')


    # DEM
    if args.dem is not None:
        heights = ('dem', args.dem)
    elif args.heightlvs is not None:
        heights = ('lvs', args.heightlvs)
    else:
        heights = ('download', None)

    # Weather
    if args.model == 'WRF':
        if args.wmnetcdf is not None:
            p.error('Argument --wmnetcdf invalid with --model WRF')
        if args.wrfmodelfiles is not None:
            weathers = {'type': 'wrf', 'files': args.wrfmodelfiles,
                        'name': 'wrf'}
        else:
            p.error('Argument --wrfmodelfiles required with --model WRF')
    elif args.model=='pickle':
        import pickle
        weathers = {'type':'pickle', 'files': args.pickleFile, 'name': 'pickle'}
    else:
        import RAiDER.models.allowed as am
        if args.model not in am.AllowedModels():
           raise NotImplementedError('Model {} is not an implemented model type'.format(args.model))
        model_module_name, model_obj = mangle_model_to_module(args.model)
        try:
            import importlib
            model_module = importlib.import_module(model_module_name)
        except ImportError:
            p.error("Couldn't find a module named {}, ".format(repr(model_module_name))+
                    "needed to load weather model {}".format(repr(args.model)))
        if args.wmnetcdf is not None:
            weathers = {'type': model_module + model_obj + '()', 'files': args.wmnetcdf,
                        'name': args.model}
        elif args.time is None:
            p.error('Must specify one of --wmnetcdf or --time (so I can '
                    'figure out what to download)')
        elif lat is None:
            p.error('Must specify one of --wmnetcdf or --area (so I can '
                    'figure out what to download)')
        else:
            weathers = {'type': model_module.(), 'files': None,
                        'name': args.model}

    if args.wmLoc is not None:
       wmLoc = args.wmLoc
    else:
       wmLoc = os.path.join(args.out, 'weather_files')

    # zref
    zref = args.zref
    
    # output file format
    if args.outformat is None:
       if args.heightlvs is not None: 
          args.outformat = 'hdf5'
       elif args.station_file is not None: 
          args.outformat = 'netcdf'
       else:
          args.outformat = 'ENVI'
   

    if args.heightlvs is not None: 
       if args.outformat.lower() != 'hdf5':
          print("WARNING: input arguments require HDF5 output file type; changing outformat to HDF5")
       outformat = 'hdf5'
    elif args.station_file is not None:
       if args.outformat.lower() != 'netcdf':
          print("WARNING: input arguments require HDF5 output file type; changing outformat to HDF5")
       outformat = 'netcdf'
    else:
       if args.outformat.lower() == 'hdf5':
          print("WARNING: output require raster output file; changing outformat to ENVI")
          outformat = 'ENVI'
       else:
          outformat = args.outformat.lower()

    # parallelization
    parallel = True if not args.no_parallel else False

    # other
    time = args.time
    out = args.out
    download_only = args.download_only
    verbose = args.verbose

    if args.area is not None:
       flag = 'files'
    elif args.bounding_box is not None:
       flag = 'bounding_box'
    elif args.station_file is not None:
       flag = 'station_file'
    else: 
       flag = None

    return los, lat, lon, heights, flag, weathers, wmLoc, zref, outformat, time, out, download_only, parallel, verbose


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
        raise NotImplemented
    return outformat

