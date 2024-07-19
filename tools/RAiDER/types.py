"""Types specific to RAiDER."""

import argparse
from dataclasses import dataclass
from typing import Literal, Optional


@dataclass
class BB:
    SNWE = tuple[float, float, float, float]

LookDir = Literal['right', 'left']

TimeInterpolationMethod = Literal['none', 'center_time', 'azimuth_time_grid']

class CalcDelaysArgsUnvalidated(argparse.Namespace):
    bucket: Optional[str]
    bucket_prefix: Optional[str]
    input_bucket_prefix: Optional[str]
    file: Optional[str]
    weather_model: str
    api_uid: Optional[str]
    api_key: Optional[str]
    interpolate_time: str
    output_directory: str

class CalcDelaysArgs(CalcDelaysArgsUnvalidated):
    bucket: Optional[str]
    bucket_prefix: Optional[str]
    input_bucket_prefix: Optional[str]
    file: str
    weather_model: str
    api_uid: Optional[str]
    api_key: Optional[str]
    interpolate_time: TimeInterpolationMethod
    output_directory: str