#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import importlib
import os

import numpy as np
import pandas as pd

from textwrap import dedent
from datetime import datetime

from RAiDER.constants import Zenith
from RAiDER.llreader import readLL
from RAiDER.ioFcns import makeDelayFileNames


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

    # Query Area
    lat, lon, llproj, bounds, flag = readLL(args.query_area)

    if (np.min(lat) < -90) | (np.max(lat) > 90):
        raise RuntimeError('Lats are out of N/S bounds; are your lat/lon coordinates switched?')

    # Line of sight calc
    if args.lineofsight is not None:
        los = ('los', args.lineofsight)
    elif args.statevectors is not None:
        los = ('sv', args.statevectors)
    else:
        los = Zenith

    # Weather
    try:
        _, model_obj = modelName2Module(args.model)
    except ModuleNotFoundError:
        raise NotImplementedError(
                dedent('''
                Model {} is not yet fully implemented, 
                please contribute!
                '''.format(args.model))
            )
    if args.model in ['WRF', 'HDF5'] and args.files is None:
        raise RuntimeError(
                'Argument --files is required with model {}'.format(args.model)
            )
    weathers = {
            'type': model_obj(), 
            'files': args.files,
            'name': args.model
        }

    # zref
    zref = args.zref

    # parallel or concurrent runs
    parallel = args.parallel
    if not parallel==1:
        import multiprocessing
        # asses the number of concurrent jobs to be executed
        max_threads = multiprocessing.cpu_count()
        if parallel == 'all':
            parallel = max_threads
        parallel = parallel if parallel < max_threads else max_threads


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
        elif flag == 'station_file':
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
            wetFilename = os.path.join(
                    out, 
                    '{}_Delay_{}_Zmax{}.csv'
                    .format(
                        args.model, 
                        time.strftime('%Y%m%dT%H%M%S'), 
                        zref
                    )
                )
            hydroFilename = wetFilename

            # copy the input file to the output location for editing
            indf = pd.read_csv(args.query_area)
            indf.to_csv(wetFilename, index=False)
        else:
            wetFilename, hydroFilename = makeDelayFileNames(
                    time, 
                    los, 
                    outformat, 
                    args.model, 
                    out
                )

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    # DEM
    if args.dem is not None:
        heights = ('dem', args.dem)
    elif args.heightlvs is not None:
        heights = ('lvs', args.heightlvs)
    elif flag == 'station_file':
        indf = pd.read_csv(args.query_area)
        try:
            hgts = indf['Hgt_m'].values
            heights = ('pandas', wetNames)
        except:
            heights = ('merge', wetNames)
    elif useWeatherNodes:
        heights = ('skip', None)
    else:
        heights = ('download', os.path.join(out, 'geom', 'warpedDEM.dem'))

    # put all the arguments in a dictionary
    outArgs = {}
    outArgs['los']=los
    outArgs['lats']=lat
    outArgs['lons']=lon
    outArgs['ll_bounds']=bounds
    outArgs['heights']=heights
    outArgs['flag']=flag
    outArgs['weather_model']=weathers
    outArgs['wmLoc']=wmLoc
    outArgs['zref']=zref
    outArgs['outformat']=outformat
    outArgs['times']=datetimeList
    outArgs['download_only']=download_only
    outArgs['out']=out
    outArgs['verbose']=verbose
    outArgs['wetFilenames']=wetNames
    outArgs['hydroFilenames']=hydroNames
    outArgs['parallel']=parallel

    return outArgs
    #return los, lat, lon, bounds, heights, flag, weathers, wmLoc, zref, outformat, datetimeList, out, download_only, verbose, wetNames, hydroNames, parallel


def modelName2Module(model_name):
    """Turn an arbitrary string into a module name.
    Takes as input a model name, which hopefully looks like ERA-I, and
    converts it to a module name, which will look like erai. I doesn't
    always produce a valid module name, but that's not the goal. The
    goal is just to handle common cases.
    Inputs:
       model_name  - Name of an allowed weather model (e.g., 'era-5')
    Outputs:
       module_name - Name of the module
       wmObject    - callable, weather model object
    """
    module_name = 'RAiDER.models.' + model_name.lower().replace('-', '')
    model_module = importlib.import_module(module_name)
    wmObject = getattr(model_module, model_name.upper().replace('-', ''))
    return module_name, wmObject


def makeDelayFileNames(
        time, 
        los, 
        outformat, 
        weather_model_name, 
        out
    ):
    '''
    Return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(time(0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    '''
    format_string = "{model_name}_{{}}_{time}{los}.{ext}".format(
        model_name=weather_model_name,
        time=time.strftime("%H_%M_%S_") if time is not None else "",
        los="ztd" if los is None else "std",
        ext=outformat
    )
    hydroname, wetname = (
        format_string.format(dtyp) for dtyp in ('hydro', 'wet')
    )

    hydro_file_name = os.path.join(out, hydroname)
    wet_file_name = os.path.join(out, wetname)
    return wet_file_name, hydro_file_name
