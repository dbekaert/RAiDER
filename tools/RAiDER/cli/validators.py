from argparse import Action, ArgumentError, ArgumentTypeError

import importlib
import itertools
import os
import re

import pandas as pd
import numpy as np

from datetime import time, timedelta, datetime, date
from textwrap import dedent
from time import strptime

from RAiDER.llreader import BoundingBox, Geocube, RasterRDR, StationFile, GeocodedFile, Geocube
from RAiDER.losreader import Zenith, Conventional
from RAiDER.utilFcns import rio_extents, rio_profile
from RAiDER.logger import logger

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
    if args.get('orbit_file'):
        if args.get('ray_trace'):
            from RAiDER.losreader import Raytracing
            los = Raytracing(args.orbit_file)
        else:
            los = Conventional(args.orbit_file)
    elif args.get('los_file'):
        if args.ray_trace:
            from RAiDER.losreader import Raytracing
            los = Raytracing(args.los_file, args.los_convention)
        else:
            los = Conventional(args.los_file, args.los_convention)

    elif args.get('los_cube'):
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
    dem_path = out

    out = {
            'dem': args.get('dem'),
            'height_file_rdr': None,
            'height_levels': None,
        }

    if args.get('dem'):
        if (station_file is not None):
            if 'Hgt_m' not in pd.read_csv(station_file):
                out['dem'] = os.path.join(dem_path, 'GLO30.dem')
        elif os.path.exists(args.dem):
            out['dem'] = args.dem
            # crop the DEM
            if bounding_box is not None:
                dem_bounds = rio_extents(rio_profile(args.dem))
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

    elif args.get('height_file_rdr'):
        out['height_file_rdr'] = args.height_file_rdr

    else:
        # download the DEM if needed
        out['dem'] = os.path.join(dem_path, 'GLO30.dem')

    if args.get('height_levels'):
        if isinstance(args.height_levels, str):
            l = re.findall('[-0-9]+', args.height_levels)
        else:
            l = args.height_levels

        out['height_levels'] = np.array([float(ll) for ll in l])
        if np.any(out['height_levels'] < 0):
            logger.warning('Weather model only extends to the surface topography; '
            'height levels below the topography will be interpolated from the surface '
            'and may be inaccurate.')

    return out


def get_query_region(args):
    '''
    Parse the query region from inputs
    '''
    # Get bounds from the inputs
    # make sure this is first
    if args.get('use_dem_latlon'):
        query = GeocodedFile(args.dem, is_dem=True)

    elif args.get('lat_file'):
        hgt_file = args.get('height_file_rdr') # only get it if exists
        dem_file = args.get('dem')
        query    = RasterRDR(args.lat_file, args.lon_file, hgt_file, dem_file)

    elif args.get('station_file'):
        query = StationFile(args.station_file)

    elif args.get('bounding_box'):
        bbox = enforce_bbox(args.bounding_box)
        if (np.min(bbox[0]) < -90) | (np.max(bbox[1]) > 90):
            raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched? Should be SNWE')
        query = BoundingBox(bbox)

    elif args.get('geocoded_file'):
        gfile  = os.path.basename(args.geocoded_file).upper()
        if (gfile.startswith('SRTM') or gfile.startswith('GLO')):
            logger.debug('Using user DEM: %s', gfile)
            is_dem = True
        else:
            is_dem = False

        query  = GeocodedFile(args.geocoded_file, is_dem=is_dem)

    ## untested
    elif args.get('geo_cube'):
        query = Geocube(args.geo_cube)

    else:
        # TODO: Need to incorporate the cube
        raise ValueError('No valid query points or bounding box found in the configuration file')


    return query


def enforce_bbox(bbox):
    """
    Enforce a valid bounding box
    """
    if isinstance(bbox, str):
        bbox = [float(d) for d in bbox.strip().split()]
    else:
        bbox = [float(d) for d in bbox]

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

    if arg_dict.get('date_list'):
        l = arg_dict['date_list']
        if isinstance(l, str):
            l = re.findall('[0-9]+', l)
        elif isinstance(l, int):
            l = [l]
        L = [enforce_valid_dates(d) for d in l]

    else:
        try:
            start = arg_dict['date_start']
        except KeyError:
            raise ValueError('Inputs must include either date_list or date_start')
        start = enforce_valid_dates(start)

        if arg_dict.get('date_end'):
            end = arg_dict['date_end']
            end = enforce_valid_dates(end)
        else:
           end = start

        if arg_dict.get('date_step'):
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


## below are for downloadGNSSDelays
def date_type(arg):
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
            return date(*strptime(arg, yf)[0:3])
        except ValueError:
            pass

    raise ArgumentTypeError(
        'Unable to coerce {} to a date. Try %Y-%m-%d'.format(arg)
    )


class MappingType(object):
    """
    A type that maps arguments to constants.

    # Example
    ```
    mapping = MappingType(foo=42, bar="baz").default(None)
    assert mapping("foo") == 42
    assert mapping("bar") == "baz"
    assert mapping("hello") is None
    ```
    """
    UNSET = object()

    def __init__(self, **kwargs):
        self.mapping = kwargs
        self._default = self.UNSET

    def default(self, default):
        """Set a default value if no mapping is found"""
        self._default = default
        return self

    def __call__(self, arg):
        if arg in self.mapping:
            return self.mapping[arg]

        if self._default is self.UNSET:
            raise KeyError(
                "Invalid choice '{}', must be one of {}".format(
                    arg, list(self.mapping.keys())
                )
            )

        return self._default


class IntegerType(object):
    """
    A type that converts arguments to integers.

    # Example
    ```
    integer = IntegerType(0, 100)
    assert integer("0") == 0
    assert integer("100") == 100
    integer("-10")  # Raises exception
    ```
    """

    def __init__(self, lo=None, hi=None):
        self.lo = lo
        self.hi = hi

    def __call__(self, arg):
        integer = int(arg)

        if self.lo is not None and integer < self.lo:
            raise ArgumentTypeError("Must be greater than {}".format(self.lo))
        if self.hi is not None and integer > self.hi:
            raise ArgumentTypeError("Must be less than {}".format(self.hi))

        return integer


class IntegerMappingType(MappingType, IntegerType):
    """
    An integer type that converts non-integer types through a mapping.

    # Example
    ```
    integer = IntegerMappingType(0, 100, random=42)
    assert integer("0") == 0
    assert integer("100") == 100
    assert integer("random") == 42
    ```
    """

    def __init__(self, lo=None, hi=None, mapping={}, **kwargs):
        IntegerType.__init__(self, lo, hi)
        kwargs.update(mapping)
        MappingType.__init__(self, **kwargs)

    def __call__(self, arg):
        try:
            return IntegerType.__call__(self, arg)
        except ValueError:
            return MappingType.__call__(self, arg)


class DateListAction(Action):
    """An Action that parses and stores a list of dates"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
    ):
        if type is not date_type:
            raise ValueError("type must be `date_type`!")

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if len(values) > 3 or not values:
            raise ArgumentError(self, "Only 1, 2 dates, or 2 dates and interval may be supplied")

        if len(values) == 2:
            start, end = values
            values = [start + timedelta(days=k) for k in range(0, (end - start).days + 1, 1)]
        elif len(values) == 3:
            start, end, stepsize = values

            if not isinstance(stepsize.day, int):
                raise ArgumentError(self, "The stepsize should be in integer days")

            new_year = date(year=stepsize.year, month=1, day=1)
            stepsize = (stepsize - new_year).days + 1

            values = [start + timedelta(days=k)
                      for k in range(0, (end - start).days + 1, stepsize)]

        setattr(namespace, self.dest, values)


class BBoxAction(Action):
    """An Action that parses and stores a valid bounding box"""

    def __init__(
        self,
        option_strings,
        dest,
        nargs=None,
        const=None,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None
    ):
        if nargs != 4:
            raise ValueError("nargs must be 4!")

        super().__init__(
            option_strings=option_strings,
            dest=dest,
            nargs=nargs,
            const=const,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar
        )

    def __call__(self, parser, namespace, values, option_string=None):
        S, N, W, E = values

        if N <= S or E <= W:
            raise ArgumentError(self, 'Bounding box has no size; make sure you use "S N W E"')

        for sn in (S, N):
            if sn < -90 or sn > 90:
                raise ArgumentError(self, 'Lats are out of S/N bounds (-90 to 90).')

        for we in (W, E):
            if we < -180 or we > 180:
                raise ArgumentError(self, 'Lons are out of W/E bounds (-180 to 180); Lons in the format of (0 to 360) are not supported.')

        setattr(namespace, self.dest, values)
