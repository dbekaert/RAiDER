import argparse
from pathlib import Path
from typing import Optional

from RAiDER.types import TimeInterpolationMethod


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

class CalcDelaysArgs(argparse.Namespace):
    bucket: Optional[str]
    bucket_prefix: Optional[str]
    input_bucket_prefix: Optional[str]
    file: Path
    weather_model: str
    api_uid: Optional[str]
    api_key: Optional[str]
    interpolate_time: TimeInterpolationMethod
    output_directory: Path
