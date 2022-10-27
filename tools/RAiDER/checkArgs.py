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
from RAiDER.llreader import bounds_from_latlon_rasters, bounds_from_csv
from RAiDER.utilFcns import rio_extents, rio_profile

_BUFFER_SIZE = 0.2 # default buffer size in lat/lon degrees 


def checkArgs(args):
    '''
    Helper fcn for checking argument compatibility and returns the
    correct variables
    '''

    #########################################################################################################################
    # Directories
    if args.weather_model_directory is None:
        args.weather_model_directory = os.path.join(args.output_directory, 'weather_files')
    if not os.path.exists(args.weather_model_directory):
        os.mkdir(args.weather_model_directory)

    #########################################################################################################################
    # Query Area
    if args.lat_file is not None:
        args.in_proj, args.bounding_box, args.pnts_file = bounds_from_latlon_rasters(args.lat_file, args.lon_file)

    elif args.station_file is not None:
        args.bounding_box, args.pnts_file, use_csv_heights = bounds_from_csv(args.station_file)
        args.in_proj = 'EPSG:4326',
         
    elif args.bounding_box is not None:
        if (np.min(args.bounding_box[0]) < -90) | (np.max(args.bounding_box[1]) > 90):
            raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched? Should be SNWE')
        args.pnts_file = '_'.join([str(int(a)) for a in [a for a in args.bounding_box][0]])
        args.ll_proj = 'EPSG:4326'

    elif args.use_dem_xy:
        try:
            args.bounding_box = rio_extents(rio_profile(args.dem))
        except FileNotFoundError:
            raise ValueError('You must specify a valid dem in the configuration file in order to use the xy points')
    else:
        raise ValueError('No valid query points or bounding box found in configuration file {}'.format(args.customTemplateFile))

    #########################################################################################################################
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
        args.los = Zenith()

    #########################################################################################################################
    # Weather model
    try:
        _, model_obj = modelName2Module(args.model)
    except ModuleNotFoundError:
        raise NotImplementedError(
            dedent('''
                Model {} is not yet fully implemented,
                please contribute!
                '''.format(args.model))
        )
    args.weather_model = model_obj()

    #TODO: This needs to get updated to account for height levels
    args.useWeatherNodes = [True if args.bounding_box is not None else False]

    #########################################################################################################################
    # Date and Time parsing
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

    #########################################################################################################################
    # filenames
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

    #########################################################################################################################
    # DEM
    if args.dem is None:
        if (args.station_file is not None) and use_csv_heights:
            args.dem_type = 'csv'
            args.dem = args.station_file
        elif args.height_file_rdr is not None:
            args.dem_type = 'hgt'
            args.dem = args.height_file_rdr
        elif args.heightlvs is not None:
            args.dem_type = 'interpolate'
        else:
            args.dem_type = 'download'
            dem_path = os.path.join(args.out, 'geom')
            args.dem = os.path.join(dem_path, 'GLO30.dem')
            if not os.path.exists(dem_path):
                os.mkdir(dem_path)
    else:
        if os.path.exists(args.dem):
            dem_bounds = rio_extents(rio_profile(args.dem))
            lats = dem_bounds[:2]
            lons = dem_bounds[2:]
            if isOutside(
                args.bounding_box,
                getBufferedExtent(
                    lats,
                    lons,
                    buf=_BUFFER_SIZE,
                )
            ):
                raise ValueError(
                    'Existing DEM does not cover the area of the input lat/lon '
                    'points; either move the DEM, delete it, or change the input '
                    'points.'
                )
            args.dem_type = 'dem'
            dem_path = os.path.dirname(args.dem)
            if not os.path.exists(dem_path):
                os.mkdir(dem_path)
        else:
            args.dem_type = 'download'
        

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
        los="ztd" if (isinstance(los, Zenith) or los is None) else "std",
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


def getBufferedExtent(lats, lons=None, buf=0.):
    '''
    get the bounding box around a set of lats/lons
    '''
    if lons is None:
        lats, lons = lats[..., 0], lons[..., 1]

    try:
        if (lats.size == 1) & (lons.size == 1):
            out = [lats - buf, lats + buf, lons - buf, lons + buf]
        elif (lats.size > 1) & (lons.size > 1):
            out = [np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)]
        elif lats.size == 1:
            out = [lats - buf, lats + buf, np.nanmin(lons), np.nanmax(lons)]
        elif lons.size == 1:
            out = [np.nanmin(lats), np.nanmax(lats), lons - buf, lons + buf]
    except AttributeError:
        if (isinstance(lats, tuple) or isinstance(lats, list)) and len(lats) == 2:
            out = [min(lats) - buf, max(lats) + buf, min(lons) - buf, max(lons) + buf]
    except Exception as e:
        raise RuntimeError('Not a valid lat/lon shape or variable')

    return np.array(out)


def isOutside(extent1, extent2):
    '''
    Determine whether any of extent1  lies outside extent2
    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon]
    Equal extents are considered "inside"
    '''
    t1 = extent1[0] < extent2[0]
    t2 = extent1[1] > extent2[1]
    t3 = extent1[2] < extent2[2]
    t4 = extent1[3] > extent2[3]
    if np.any([t1, t2, t3, t4]):
        return True
    return False


def isInside(extent1, extent2):
    '''
    Determine whether all of extent1 lies inside extent2
    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon].
    Equal extents are considered "inside"
    '''
    t1 = extent1[0] <= extent2[0]
    t2 = extent1[1] >= extent2[1]
    t3 = extent1[2] <= extent2[2]
    t4 = extent1[3] >= extent2[3]
    if np.all([t1, t2, t3, t4]):
        return True
    return False
