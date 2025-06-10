"""Polyfills for several symbols used for types that rasterio doesn't export."""

from dataclasses import dataclass
from typing import TypedDict, Union

import rasterio.crs
import rasterio.transform


GDAL = tuple[float, float, float, float, float, float]

@dataclass
class Statistics:
    max: float
    mean: float
    min: float
    std: float


class Profile(TypedDict):
    driver: str
    width: int
    height: int
    count: int
    crs: Union[str, dict, rasterio.crs.CRS]
    transform: rasterio.transform.Affine
    dtype: str
