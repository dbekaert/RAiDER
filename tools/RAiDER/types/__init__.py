"""Types specific to RAiDER."""

import argparse
from pathlib import Path
from typing import Literal, Optional, TypedDict, Union

import rasterio.crs
import rasterio.transform


LookDir = Literal['right', 'left']

TimeInterpolationMethod = Literal['none', 'center_time', 'azimuth_time_grid']

class CalcDelaysArgsUnparsed(argparse.Namespace):
    bucket: Optional[str]
    bucket_prefix: Optional[str]
    input_bucket_prefix: Optional[str]
    file: Optional[Path]
    weather_model: str
    api_uid: Optional[str]
    api_key: Optional[str]
    interpolate_time: TimeInterpolationMethod
    output_directory: Path

class CalcDelaysArgs(CalcDelaysArgsUnparsed):
    file: Path

class RIOProfile(TypedDict):
    driver: str
    width: int
    height: int
    count: int
    crs: Union[str, dict, rasterio.crs.CRS]
    transform: rasterio.transform.Affine
    dtype: str
