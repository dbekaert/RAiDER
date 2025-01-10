import argparse
import datetime as dt
import importlib
import re
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd


if sys.version_info >= (3,11):
    from typing import Self
else:
    Self = Any

from RAiDER.cli.types import (
    AOIGroupUnparsed,
    DateGroup,
    DateGroupUnparsed,
    HeightGroup,
    HeightGroupUnparsed,
    LOSGroupUnparsed,
    RuntimeGroup,
)
from RAiDER.llreader import AOI, BoundingBox, GeocodedFile, Geocube, RasterRDR, StationFile
from RAiDER.logger import logger
from RAiDER.losreader import LOS, Conventional, Zenith
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.types import BB
from RAiDER.utilFcns import rio_extents, rio_profile


_BUFFER_SIZE = 0.2  # default buffer size in lat/lon degrees


def parse_weather_model(weather_model_name: str, aoi: AOI) -> WeatherModel:
    weather_model_name = weather_model_name.upper().replace('-', '')
    try:
        _, Model = get_wm_by_name(weather_model_name)
    except ModuleNotFoundError:
        if weather_model_name == 'AUTO':
            _, Model = get_wm_by_name('HRRR')
        raise NotImplementedError(
            f'Model {weather_model_name} is not yet fully implemented, please contribute!'
        )

    # Check that the user-requested bounding box is within the weather model domain
    model: WeatherModel = Model()
    model.checkValidBounds(aoi.bounds())

    return model


def get_los(los_group: LOSGroupUnparsed) -> LOS:
    if los_group.orbit_file is not None:
        if los_group.ray_trace:
            from RAiDER.losreader import Raytracing
            los = Raytracing(los_group.orbit_file)
        else:
            los = Conventional(los_group.orbit_file)

    elif los_group.los_file is not None:
        if los_group.ray_trace:
            from RAiDER.losreader import Raytracing
            los = Raytracing(los_group.los_file, los_group.los_convention)
        else:
            los = Conventional(los_group.los_file, los_group.los_convention)

    elif los_group.los_cube is not None:
        raise NotImplementedError('LOS_cube is not yet implemented')
        # if los_group.ray_trace:
        #     los = Raytracing(los_group.los_cube)
        # else:
        #     los = Conventional(los_group.los_cube)
    else:
        los = Zenith()

    return los


def get_heights(height_group: HeightGroupUnparsed, aoi_group: AOIGroupUnparsed, runtime_group: RuntimeGroup) -> HeightGroup:
    """Parse the Height info and download a DEM if needed."""
    result = HeightGroup(
        dem=height_group.dem,
        use_dem_latlon=height_group.use_dem_latlon,
        height_file_rdr=height_group.height_file_rdr,
        height_levels=None,
    )

    if height_group.dem is not None:
        if aoi_group.station_file is not None:
            station_data = pd.read_csv(aoi_group.station_file)
            if 'Hgt_m' not in station_data:
                result.dem = runtime_group.output_directory / 'GLO30.dem'
        elif Path(height_group.dem).exists():
            # crop the DEM
            if aoi_group.bounding_box is not None:
                dem_bounds = rio_extents(rio_profile(height_group.dem))
                lats: BB.SN = dem_bounds[:2]
                lons: BB.WE = dem_bounds[2:]
                if isOutside(
                    parse_bbox(aoi_group.bounding_box),
                    getBufferedExtent(
                        lats,
                        lons,
                        buffer_size=_BUFFER_SIZE,
                    ),
                ):
                    raise ValueError(
                        'Existing DEM does not cover the area of the input lat/lon points; either move the DEM, delete '
                        'it, or change the input points.'
                    )
        # else: will download the dem later

    elif height_group.height_file_rdr is None:
        # download the DEM if needed
        result.dem = runtime_group.output_directory / 'GLO30.dem'

    if height_group.height_levels is not None:
        if isinstance(height_group.height_levels, str):
            levels = re.findall('[-0-9]+', height_group.height_levels)
        else:
            levels = height_group.height_levels

        levels = np.array([float(level) for level in levels])
        if np.any(levels < 0):
            logger.warning(
                'Weather model only extends to the surface topography; '
                'height levels below the topography will be interpolated from the surface and may be inaccurate.'
            )
        result.height_levels = list(levels)

    return result


def get_query_region(aoi_group: AOIGroupUnparsed, height_group: HeightGroupUnparsed, cube_spacing_in_m: float) -> AOI:
    """Parse the query region from inputs.
    
    This function determines the query region from the input parameters. It will return an AOI object that can be used
    to query the weather model.
    Note: both an AOI group and a height group are necessary in case a DEM is needed.
    """
    # Get bounds from the inputs
    # make sure this is first
    if height_group.use_dem_latlon:
        query = GeocodedFile(Path(height_group.dem), is_dem=True, cube_spacing_in_m=cube_spacing_in_m)

    elif aoi_group.lat_file is not None or aoi_group.lon_file is not None:
        if aoi_group.lat_file is None or aoi_group.lon_file is None:
            raise ValueError('A lon_file must be specified if a lat_file is specified')
        query = RasterRDR(
            aoi_group.lat_file, aoi_group.lon_file,
            height_group.height_file_rdr, height_group.dem,
            cube_spacing_in_m=cube_spacing_in_m
        )

    elif aoi_group.station_file is not None:
        query = StationFile(aoi_group.station_file, cube_spacing_in_m=cube_spacing_in_m)

    elif aoi_group.bounding_box is not None:
        bbox = parse_bbox(aoi_group.bounding_box)
        if np.min(bbox[0]) < -90 or np.max(bbox[1]) > 90:
            raise ValueError('Lats are out of N/S bounds; are your lat/lon coordinates switched? Should be SNWE')
        query = BoundingBox(bbox, cube_spacing_in_m=cube_spacing_in_m)

    elif aoi_group.geocoded_file is not None:
        geocoded_file_path = Path(aoi_group.geocoded_file)
        filename = geocoded_file_path.name.upper()
        if filename.startswith('SRTM') or filename.startswith('GLO'):
            logger.debug('Using user DEM: %s', filename)
            is_dem = True
        else:
            is_dem = False
        query = GeocodedFile(geocoded_file_path, is_dem=is_dem, cube_spacing_in_m=cube_spacing_in_m)

    # untested
    elif aoi_group.geo_cube is not None:
        query = Geocube(aoi_group.geo_cube, cube_spacing_in_m)

    else:
        # TODO: Need to incorporate the cube
        raise ValueError('No valid query points or bounding box found in the configuration file')

    return query


def parse_bbox(bbox: Union[str, list[Union[int, float]], tuple]) -> BB.SNWE:
    """Parse a bounding box string input and ensure it is valid."""
    if isinstance(bbox, str):
        bbox = [float(d) for d in bbox.strip().split()]
    else:
        bbox = [float(d) for d in bbox]

    # Check the bbox
    if len(bbox) != 4:
        raise ValueError('bounding box must have 4 elements!')
    S, N, W, E = bbox

    if N <= S or E <= W:
        raise ValueError('Bounding box has no size; make sure you use "S N W E"')

    for sn in (S, N):
        if sn < -90 or sn > 90:
            raise ValueError('Lats are out of S/N bounds (-90 to 90).')

    for we in (W, E):
        if we < -180 or we > 180:
            raise ValueError(
                'Lons are out of W/E bounds (-180 to 180); Lons in the format of (0 to 360) are not supported.'
            )

    return S, N, W, E


def parse_dates(date_group: DateGroupUnparsed) -> DateGroup:
    """Determine the requested dates from the input parameters."""
    if date_group.date_list is not None:
        if isinstance(date_group.date_list, str):
            unparsed_dates = re.findall('[0-9]+', date_group.date_list)
        elif isinstance(date_group.date_list, int):
            unparsed_dates = [date_group.date_list]
        else:
            unparsed_dates = date_group.date_list
        date_list = [coerce_into_date(d) for d in unparsed_dates]

    else:
        if date_group.date_start is None:
            raise ValueError('Inputs must include either date_list or date_start')
        start = coerce_into_date(date_group.date_start)

        if date_group.date_end is not None:
            end = coerce_into_date(date_group.date_end)
        else:
            end = start

        if date_group.date_step:
            step = int(date_group.date_step)
        else:
            step = 1

        date_list = [
            start + dt.timedelta(days=step)
            for step in range(0, (end - start).days + 1, step)
        ]
    
    return DateGroup(
        date_list=date_list,
    )


def coerce_into_date(val: Union[int, str]) -> dt.date:
    """Parse a date from a string in pseudo-ISO 8601 format."""
    year_formats = (
        '%Y-%m-%d',
        '%Y%m%d',
        '%d',
        '%j',
    )

    for yf in year_formats:
        try:
            return dt.datetime.strptime(str(val), yf).date()
        except ValueError:
            pass

    raise ValueError(f'Unable to coerce {val} to a date. Try %Y-%m-%d')


def get_wm_by_name(model_name: str) -> tuple[str, WeatherModel]:
    """
    Turn an arbitrary string into a module name.

    Takes as input a model name, which hopefully looks like ERA-I, and
    converts it to a module name, which will look like erai. It doesn't
    always produce a valid module name, but that's not the goal. The
    goal is just to handle common cases.
    Inputs:
       model_name  - Name of an allowed weather model (e.g., 'era-5')
    Outputs:
       module_name - Name of the module
       wmObject    - callable, weather model object.
    """
    module_name = 'RAiDER.models.' + model_name.lower().replace('-', '')
    module = importlib.import_module(module_name)
    Model = getattr(module, model_name.upper().replace('-', ''))
    return module_name, Model


def getBufferedExtent(lats: BB.SN, lons: BB.WE, buffer_size: float=0.0) -> BB.SNWE:
    """Get the bounding box around a set of lats/lons."""
    return (
        min(lats) - buffer_size,
        max(lats) + buffer_size,
        min(lons) - buffer_size,
        max(lons) + buffer_size
    )


def isOutside(extent1: BB.SNWE, extent2: BB.SNWE) -> bool:
    """Determine whether any of extent1 lies outside extent2.

    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon] (SNWE).
    Equal extents are considered "inside".
    """
    t1 = extent1[0] < extent2[0]
    t2 = extent1[1] > extent2[1]
    t3 = extent1[2] < extent2[2]
    t4 = extent1[3] > extent2[3]
    return any((t1, t2, t3, t4))


def isInside(extent1: BB.SNWE, extent2: BB.SNWE) -> bool:
    """Determine whether all of extent1 lies inside extent2.

    extent1/2 should be a list containing [lower_lat, upper_lat, left_lon, right_lon] (SNWE).
    Equal extents are considered "inside".
    """
    t1 = extent1[0] <= extent2[0]
    t2 = extent1[1] >= extent2[1]
    t3 = extent1[2] <= extent2[2]
    t4 = extent1[3] >= extent2[3]
    return all((t1, t2, t3, t4))


# below are for downloadGNSSDelays
def date_type(val: Union[int, str]) -> dt.date:
    """Parse a date from a string in pseudo-ISO 8601 format."""
    try:
        return coerce_into_date(val)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc))


class MappingType:
    """A type that maps arguments to constants.

    # Example
    ```
    mapping = MappingType(foo=42, bar="baz").default(None)
    assert mapping("foo") == 42
    assert mapping("bar") == "baz"
    assert mapping("hello") is None
    ```
    """

    UNSET = object()
    _default: Union[object, Any]

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        self.mapping = kwargs
        self._default = self.UNSET

    def default(self, default: Any) -> Self:  # noqa: ANN401
        """Set a default value if no mapping is found."""
        self._default = default
        return self

    def __call__(self, arg: str) -> Any:  # noqa: ANN401
        if arg in self.mapping:
            return self.mapping[arg]

        if self._default is self.UNSET:
            raise KeyError(f"Invalid choice '{arg}', must be one of {list(self.mapping.keys())}")

        return self._default


class IntegerOnRangeType:
    """A type that converts arguments to integers and enforces that they are on a certain range.

    # Example
    ```
    integer = IntegerType(0, 100)
    assert integer("0") == 0
    assert integer("100") == 100
    integer("-10")  # Raises exception
    ```
    """

    def __init__(self, lo: Optional[int]=None, hi: Optional[int]=None) -> None:
        self.lo = lo
        self.hi = hi

    def __call__(self, arg: Any) -> int:  # noqa: ANN401
        integer = int(arg)

        if self.lo is not None and integer < self.lo:
            raise argparse.ArgumentTypeError(f'Must be greater than {self.lo}')
        if self.hi is not None and integer > self.hi:
            raise argparse.ArgumentTypeError(f'Must be less than {self.hi}')

        return integer


class IntegerMappingType(MappingType, IntegerOnRangeType):
    """An integer type that converts non-integer types through a mapping.

    # Example
    ```
    integer = IntegerMappingType(0, 100, random=42)
    assert integer("0") == 0
    assert integer("100") == 100
    assert integer("random") == 42
    ```
    """

    def __init__(self, lo: Optional[int]=None, hi: Optional[int]=None, mapping: Optional[dict[str, Any]]={}, **kwargs: dict[str, Any]) -> None:
        IntegerOnRangeType.__init__(self, lo, hi)
        kwargs.update(mapping)
        MappingType.__init__(self, **kwargs)

    def __call__(self, arg: Any) -> Union[int, Any]:  # noqa: ANN401
        try:
            return IntegerOnRangeType.__call__(self, arg)
        except ValueError:
            return MappingType.__call__(self, arg)


class DateListAction(argparse.Action):
    """An Action that parses and stores a list of dates."""

    def __init__(
        self,
        option_strings,  # noqa: ANN001 -- see argparse.Action.__init__
        dest,  # noqa: ANN001
        nargs=None,  # noqa: ANN001
        const=None,  # noqa: ANN001
        default=None,  # noqa: ANN001
        type=None,  # noqa: ANN001
        choices=None,  # noqa: ANN001
        required=False,  # noqa: ANN001
        help=None,  # noqa: ANN001
        metavar=None,  # noqa: ANN001
    ) -> None:
        if type is not date_type:
            raise ValueError('type must be `date_type`!')

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
            metavar=metavar,
        )

    def __call__(self, _, namespace, values, __=None):  # noqa: ANN001, ANN204 -- see argparse.Action.__call__
        if len(values) > 3 or not values:
            raise argparse.ArgumentError(self, 'Only 1, 2 dates, or 2 dates and interval may be supplied')

        if len(values) == 2:
            start, end = values
            values = [start + dt.timedelta(days=k) for k in range(0, (end - start).days + 1, 1)]
        elif len(values) == 3:
            start, end, stepsize = values

            if not isinstance(stepsize.day, int):
                raise argparse.ArgumentError(self, 'The stepsize should be in integer days')

            new_year = dt.date(year=stepsize.year, month=1, day=1)
            stepsize = (stepsize - new_year).days + 1

            values = [start + dt.timedelta(days=k) for k in range(0, (end - start).days + 1, stepsize)]

        setattr(namespace, self.dest, values)


class BBoxAction(argparse.Action):
    """An Action that parses and stores a valid bounding box."""

    def __init__(
        self,
        option_strings,  # noqa: ANN001 -- see argparse.Action.__init__
        dest,  # noqa: ANN001
        nargs=None,  # noqa: ANN001
        const=None,  # noqa: ANN001
        default=None,  # noqa: ANN001
        type=None,  # noqa: ANN001
        choices=None,  # noqa: ANN001
        required=False,  # noqa: ANN001
        help=None,  # noqa: ANN001
        metavar=None,  # noqa: ANN001
    ) -> None:
        if nargs != 4:
            raise ValueError('nargs must be 4!')

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
            metavar=metavar,
        )

    def __call__(self, _, namespace, values, __=None):  # noqa: ANN001, ANN204 -- see argparse.Action.__call__
        S, N, W, E = values

        if N <= S or E <= W:
            raise argparse.ArgumentError(self, 'Bounding box has no size; make sure you use "S N W E"')

        for sn in (S, N):
            if sn < -90 or sn > 90:
                raise argparse.ArgumentError(self, 'Lats are out of S/N bounds (-90 to 90).')

        for we in (W, E):
            if we < -180 or we > 180:
                raise argparse.ArgumentError(
                    self,
                    'Lons are out of W/E bounds (-180 to 180); Lons in the format of (0 to 360) are not supported.',
                )

        setattr(namespace, self.dest, values)
