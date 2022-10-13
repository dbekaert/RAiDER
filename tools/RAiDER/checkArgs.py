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
from datetime import datetime, date, time, timedelta

from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.llreader import readLL


def checkArgs(args):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''
    if args.weather_model_directory is None:
        args.weather_model_directory = os.path.join(args.output_directory, 'weather_files')
    if not os.path.exists(args.weather_model_directory):
        os.mkdir(args.weather_model_directory)

    # Query Area
    if args.lat_file is not None:
        pass
        # lat, lon, llproj, bounds, flag, pnts_file = readLL(args.query_area)
    elif args.station_file is not None:
        pass
    elif args.bounding_box is not None:
        lat, lon = None, None # TODO
        if (np.min(lat) < -90) | (np.max(lat) > 90):
            raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched?')
    elif args.use_dem_xy:
        pass
    else:
        raise ValueError('No valid query points or bounding box found in configuration file {}'.format(args.customTemplateFile))

    # Line of sight 
    if args.orbit_file is not None:
        args.los = Conventional(args.los_file)
    elif args.los_file is not None:
        if args.ray_trace:
            args.los = Raytracing(args.orbit_file)
        else:
            args.los = Conventional(args.orbit_file)
    elif args.los_cube is not None:
        if args.ray_trace:
            args.los = Raytracing(args.los_cube)
        else:
            args.los = Conventional(args.los_cube)
    else:
        los = Zenith()

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

    # handle the datetimes requested
    args.time = time(args.time)

    if (args.date_start is None) and (args.date_list is None):
        raise ValueError('You must specify either a date_list or date_start in the configuration file')
    elif args.date_start is not None:
        args.date_start = date(args.date_start[:4], args.date_start[4:6], args.date_start[6:])
    else:
        args.date_list = [date(d[:4], d[4:6], d[6:]) for d in args.date_list]
    if args.date_end is not None:
        args.date_end = date(args.date_end[:4], args.date_end[4:6], args.date_end[6:])
    else:
        args.date_end = args.date_start
    if args.date_list is None:
        if args.date_end is not None:
            if args.date_inc is None:
                args.date_inc = 1
        args.date_list = [args.date_start + timedelta(days=args.date_inc) for k in range(0, (args.date_end - args.date_start).days + 1, 1)]
    args.date_list = [datetime.combine(d, args.time) for d in args.date_list]

    # Initiate the weather model object
    args.weather_model = model_obj()
    args.useWeatherNodes = [True if args.bounding_box is not None else False]

    wetNames, hydroNames = [], []
    for time in args.date_list:
        if args.station_file is not None:
            wetFilename = os.path.join(
                args.out,
                '{}_Delay_{}.csv'
                .format(
                    args.model,
                    args.time.strftime('%Y%m%dT%H%M%S'),
                )
            )
            hydroFilename = wetFilename

            # copy the input file to the output location for editing
            indf = pd.read_csv(args.query_area).drop_duplicates(subset=["Lat", "Lon"])
            indf.to_csv(wetFilename, index=False)
        else:
            wetFilename, hydroFilename = makeDelayFileNames(
                args.time,
                args.los,
                args.raster_format,
                args.model,
                args.out,
            )

        wetNames.append(wetFilename)
        hydroNames.append(hydroFilename)

    # DEM
    if (args.heightlvs is None) and (args.dem is None):
        args.dem = 'Download'
        if args.station_file is not None:
            df = pd.read_csv(args.station_file)
            if 'Hgt_m' in df.columns:
                args.dem = None
    if args.dem == 'Download':
        args.dem_path = os.path.join(args.out, 'geom')
        if not os.path.exists(args.dem_path):
            os.mkdir(args.dem_path)

    return args


def makeDelayFileNames(time, los, outformat, weather_model_name, out):
    '''
    return names for the wet and hydrostatic delays.

    # Examples:
    >>> makeDelayFileNames(datetime(2020, 1, 1, 0, 0, 0), None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_00_00_00_ztd.h5', 'some_dir/model_name_hydro_00_00_00_ztd.h5')
    >>> makeDelayFileNames(None, None, "h5", "model_name", "some_dir")
    ('some_dir/model_name_wet_ztd.h5', 'some_dir/model_name_hydro_ztd.h5')
    '''
    format_string = "{model_name}_{{}}_{time}{los}.{ext}".format(
        model_name=weather_model_name,
        time=time.strftime("%Y%m%dT%H%M%S_") if time is not None else "",
        los=["ztd" if los==Zenith else 'std'],
        ext=outformat
    )
    hydroname, wetname = (
        format_string.format(dtyp) for dtyp in ('hydro', 'wet')
    )

    hydro_file_name = os.path.join(out, hydroname)
    wet_file_name = os.path.join(out, wetname)
    return wet_file_name, hydro_file_name


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
