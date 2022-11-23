import importlib
import itertools
import os

import pandas as pd
import numpy as np

from datetime import time, timedelta, datetime
from textwrap import dedent
from time import strptime

from RAiDER.llreader import BoundingBox, Geocube, RasterRDR, StationFile, GeocodedFile, Geocube
from RAiDER.losreader import Zenith, Conventional, Raytracing
from RAiDER.utilFcns import rio_extents, rio_profile

_BUFFER_SIZE = 0.2 # default buffer size in lat/lon degrees

def enforce_wm(value):
    model = value.upper().replace("-", "")
    try:
        _, model_obj = modelName2Module(model)
    except ModuleNotFoundError:
        raise NotImplementedError(
            dedent('''
                Model {} is not yet fully implemented,
                please contribute!
                '''.format(model))
        )
    return model_obj()


def get_los(args):
    if 'orbit_file' in args.keys():
        if args.ray_trace:
            los = Raytracing(args.orbit_file)
        else:
            los = Conventional(args.orbit_file)
    elif 'los_file' in args.keys():
        if args.ray_trace:
            los = Raytracing(args.los_file, args.los_convention)
        else:
            los = Conventional(args.los_file, args.los_convention)
    elif 'los_cube' in args.keys():
        raise NotImplementedError('LOS_cube is not yet implemented')
#        if args.ray_trace:
#            los = Raytracing(args.los_cube)
#        else:
#            los = Conventional(args.los_cube)
    else:
        los = Zenith()

    return los


def get_heights(args, out, station_file, bounding_box=None):
    '''
    Parse the Height info and download a DEM if needed
    '''
    dem_path = os.path.join(out, 'geom')
    if not os.path.exists(dem_path):
        os.mkdir(dem_path)
    out = {
            'dem': None,
            'height_file_rdr': None,
            'height_levels': None,
        }

    if 'dem' in args.keys():
        if (station_file is not None):
            if 'Hgt_m' not in pd.read_csv(station_file):
                out['dem'] = os.path.join(dem_path, 'GLO30.dem')
        elif os.path.exists(args.dem):
            out['dem'] = args['dem']
            if bounding_box is not None:
                dem_bounds = rio_extents(rio_profile(args.dem)[0])
                lats = dem_bounds[:2]
                lons = dem_bounds[2:]
                if isOutside(
                    bounding_box,
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
        else:
            pass # will download the dem later

    elif 'height_file_rdr' in args.keys():
        out['height_file_rdr'] = args.height_file_rdr

    elif 'height_levels' in args.keys():
        out['height_levels'] = [float(l) for l in args.height_levels.strip().split()]

    else:
        # download the DEM if needed
        out['dem'] = os.path.join(dem_path, 'GLO30.dem')

    return out


def get_query_region(args):
    '''
    Parse the query region from inputs
    '''
    # Get bounds from the inputs
    # make sure this is first
    if 'use_dem_latlon' in args.keys():
        query = GeocodedFile(args.dem, is_dem=True)

    elif 'lat_file' in args.keys():
        hgt_file = args.get('hgt_file_rdr', None) # only get it if exists
        query    = RasterRDR(args.lat_file, args.lon_file, hgt_file)

    elif 'station_file' in args.keys():
        query = StationFile(args.station_file)

    elif 'bounding_box' in args.keys():
        bbox = enforce_bbox(args.bounding_box)
        if (np.min(bbox[0]) < -90) | (np.max(bbox[1]) > 90):
            raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched? Should be SNWE')
        query = BoundingBox(bbox)

    elif 'geocoded_file' in args.keys():
        query = GeocodedFile(args.geocoded_file, is_dem=False)

    ## untested
    elif 'los_cube' in args.keys():
        query = Geocube(args.los_cube)

    else:
        # TODO: Need to incorporate the cube
        raise ValueError('No valid query points or bounding box found in the configuration file')


    return query


def enforce_bbox(bbox):
    """
    Enforce a valid bounding box
    """
    bbox = [float(d) for d in bbox.strip().split()]

    # Check the bbox
    if len(bbox) != 4:
        raise ValueError("bounding box must have 4 elements!")
    S, N, W, E = bbox

    if N <= S or E <= W:
        raise ValueError('Bounding box has no size; make sure you use "S N W E"')

    for sn in (S, N):
        if sn < -90 or sn > 90:
            raise ValueError('Lats are out of S/N bounds (-90 to 90).')

    for we in (W, E):
        if we < -180 or we > 180:
            raise ValueError('Lons are out of W/E bounds (-180 to 180); Lons in the format of (0 to 360) are not supported.')

    return bbox


def parse_dates(arg_dict):
    '''
    Determine the requested dates from the input parameters
    '''

    if 'date_list' in arg_dict.keys():
        l = arg_dict['date_list']
        L = [enforce_valid_dates(d) for d in l]

    else:
        try:
            start = arg_dict['date_start']
        except KeyError:
            raise ValueError('Inputs must include either date_list or date_start')
        start = enforce_valid_dates(start)

        if 'date_end' in arg_dict.keys():
            end = arg_dict['date_end']
            end = enforce_valid_dates(end)
        else:
           end = start

        if 'date_step' in arg_dict.keys():
            step = int(arg_dict['date_step'])
        else:
            step = 1

        L = [start + timedelta(days=step) for step in range(0, (end - start).days + 1, step)]

    return L


def enforce_valid_dates(arg):
    """
    Parse a date from a string in pseudo-ISO 8601 format.
    """
    year_formats = (
        '%Y-%m-%d',
        '%Y%m%d',
        '%d',
        '%j',
    )

    for yf in year_formats:
        try:
            return datetime.strptime(str(arg), yf)
        except ValueError:
            pass


    raise ValueError(
        'Unable to coerce {} to a date. Try %Y-%m-%d'.format(arg)
    )


def enforce_time(arg_dict):
    '''
    Parse an input time (required to be ISO 8601)
    '''
    try:
        arg_dict['time'] = convert_time(arg_dict['time'])
    except KeyError:
        raise ValueError('You must specify a "time" in the input config file')

    if 'end_time' in arg_dict.keys():
        arg_dict['end_time'] = convert_time(arg_dict['end_time'])
    return arg_dict


def convert_time(inp):
    time_formats = (
        '',
        'T%H:%M:%S.%f',
        'T%H%M%S.%f',
        '%H%M%S.%f',
        'T%H:%M:%S',
        '%H:%M:%S',
        'T%H%M%S',
        '%H%M%S',
        'T%H:%M',
        'T%H%M',
        '%H:%M',
        'T%H',
    )
    timezone_formats = (
        '',
        'Z',
        '%z',
    )
    all_formats = map(
        ''.join,
        itertools.product(time_formats, timezone_formats)
    )

    for tf in all_formats:
        try:
            return time(*strptime(inp, tf)[3:6])
        except ValueError:
            pass

    raise ValueError(
                'Unable to coerce {} to a time.'+
                'Try T%H:%M:%S'.format(inp)
        )


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
