import argparse
import dataclasses
import datetime as dt
import itertools
import time
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np

from RAiDER.constants import _CUBE_SPACING_IN_M
from RAiDER.llreader import AOI
from RAiDER.losreader import LOS
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.types import BB, LookDir, TimeInterpolationMethod


LOSConvention = Literal['isce', 'hyp3']

@dataclasses.dataclass
class DateGroupUnparsed:
    date_start: Optional[Union[int, str]] = None
    date_end: Optional[Union[int, str]] = None
    date_step: Optional[Union[int, str]] = None
    date_list: Optional[Union[int, str]] = None

@dataclasses.dataclass
class DateGroup:
    # After the dates have been parsed, only the date list is valid, and all
    # other fields of the date group should not be used.
    date_list: list[dt.date]


class TimeGroup:
    """Parse an input time (required to be ISO 8601)."""
    _DEFAULT_ACQUISITION_WINDOW_SEC = 30
    TIME_FORMATS = (
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
    TIMEZONE_FORMATS = (
        '',
        'Z',
        '%z',
    )
    time: dt.time
    end_time: dt.time
    interpolate_time: Optional[TimeInterpolationMethod]

    def __init__(
        self,
        time: Optional[Union[str, dt.time]] = None,
        end_time: Optional[Union[str, dt.time]] = None,
        interpolate_time: Optional[TimeInterpolationMethod] = None,
    ) -> None:
        self.interpolate_time = interpolate_time

        if time is None:
            raise ValueError('You must specify a "time" in the input config file')
        if isinstance(time, dt.time):
            self.time = time
        else:
            self.time = TimeGroup.coerce_into_time(time)

        if end_time is not None:
            if isinstance(end_time, dt.time):
                self.end_time = end_time
            else:
                self.end_time = TimeGroup.coerce_into_time(end_time)
            if self.end_time < self.time:
                raise ValueError(
                    'Acquisition start time must be before end time. '
                    f'Provided start time {self.time} is later than end time {self.end_time}'
                )
        else:
            sentinel_datetime = dt.datetime.combine(dt.date(1900, 1, 1), self.time)
            new_end_time = sentinel_datetime + dt.timedelta(seconds=TimeGroup._DEFAULT_ACQUISITION_WINDOW_SEC)
            self.end_time = new_end_time.time()
            if self.end_time < self.time:
                raise ValueError(
                    'Acquisition start time must be before end time. '
                    f'Provided start time {self.time} is later than end time {self.end_time} '
                    f'(with default window of {TimeGroup._DEFAULT_ACQUISITION_WINDOW_SEC} seconds)'
                )
    
    @staticmethod
    def coerce_into_time(val: Union[int, str]) -> dt.time:
        val = str(val)
        all_formats = map(''.join, itertools.product(TimeGroup.TIME_FORMATS, TimeGroup.TIMEZONE_FORMATS))
        for tf in all_formats:
            try:
                return dt.time(*time.strptime(val, tf)[3:6])
            except ValueError:
                pass
        raise ValueError(f'Unable to coerce "{val}" to a time. Try T%H:%M:%S')

@dataclasses.dataclass
class AOIGroupUnparsed:
    bounding_box: Optional[Union[str, list[Union[float, int]], BB.SNWE]] = None
    geocoded_file: Optional[str] = None
    lat_file: Optional[str] = None
    lon_file: Optional[str] = None
    station_file: Optional[str] = None
    geo_cube: Optional[str] = None

@dataclasses.dataclass
class AOIGroup:
    # Once the AOI group is parsed, the members from the config file should not
    # be read again. Instead, the parsed AOI will be available on AOIGroup.aoi.
    aoi: AOI


@dataclasses.dataclass
class HeightGroupUnparsed:
    dem: Optional[str] = None
    use_dem_latlon: bool = False
    height_file_rdr: Optional[str] = None
    height_levels: Optional[Union[str, list[Union[float, int]]]] = None

@dataclasses.dataclass
class HeightGroup:
    dem: Optional[str]
    use_dem_latlon: bool
    height_file_rdr: Optional[str]
    height_levels: Optional[list[float]]


@dataclasses.dataclass
class LOSGroupUnparsed:
    ray_trace: bool = False
    los_file: Optional[str] = None
    los_convention: LOSConvention = 'isce'
    los_cube: Optional[str] = None
    orbit_file: Optional[str] = None
    zref: Optional[np.float64] = None

@dataclasses.dataclass
class LOSGroup:
    los: LOS
    ray_trace: bool = False
    los_file: Optional[str] = None
    los_convention: LOSConvention = 'isce'
    los_cube: Optional[str] = None
    orbit_file: Optional[str] = None
    zref: Optional[np.float64] = None

class RuntimeGroup:
    raster_format: str
    file_format: str  # TODO(garlic-os): redundant with raster_format?
    verbose: bool
    output_projection: str
    cube_spacing_in_m: float
    download_only: bool
    output_directory: Path
    weather_model_directory: Path

    def __init__(
        self,
        raster_format: str = 'GTiff',
        file_format: str = 'GTiff',
        verbose: bool = True,
        output_projection: str = 'EPSG:4326',
        cube_spacing_in_m: float = _CUBE_SPACING_IN_M,
        download_only: bool = False,
        output_directory: str = '.',
        weather_model_directory: Optional[str] = None,
    ):
        self.raster_format = raster_format
        self.file_format = file_format
        self.verbose = verbose
        self.output_projection = output_projection
        self.cube_spacing_in_m = cube_spacing_in_m
        self.download_only = download_only
        self.output_directory = Path(output_directory)
        if weather_model_directory is not None:
            self.weather_model_directory = Path(weather_model_directory)
        else:
            self.weather_model_directory = self.output_directory / 'weather_files'


@dataclasses.dataclass
class RunConfig:
    weather_model: WeatherModel
    date_group: DateGroup
    time_group: TimeGroup
    aoi_group: AOIGroup
    height_group: HeightGroup
    los_group: LOSGroup
    runtime_group: RuntimeGroup
    look_dir: LookDir = 'right'
    cube_spacing_in_m: Optional[float] = None  # deprecated
    wetFilenames: Optional[list[str]] = None
    hydroFilenames: Optional[list[str]] = None


class RAiDERArgs(argparse.Namespace):
    download_only: bool = False
    generate_config: Optional[str] = None
    run_config_file: Optional[Path]
