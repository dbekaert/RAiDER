from typing import Any, List, Optional, Tuple, TypedDict
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from pathlib import Path
import datetime as dt
import numpy as np
from RAiDER.models.weatherModel import WeatherModel
from RAiDER.losreader import LOS

Lats = np.array  # length: 2
Lons = np.array  # length: 2
LLBounds = Tuple[float, float, float, float]
Heights = Tuple[
    Literal["dem", "lvs", "pandas", "merge", "download", "skip"],
    Optional[Any]  # TODO: type
]
Flag = Literal["files", "bounding_box", "station_file"]
OutFormat = Literal["envi", "csv", "hdf5"]


class WeatherDict(TypedDict):
    type: WeatherModel
    files: Optional[List[Path]]
    name: Literal["era5", "era5t", "erai", "merra2", "wrf", "hrrr", "gmao",
                  "hdf5", "hres", "ncmr"]


class Arguments(TypedDict):
    los: Optional[LOS]
    lats: Lats
    lons: Lons
    ll_bounds: LLBounds
    heights: Heights
    flag: Flag
    weather_model: WeatherDict
    wmLoc: Path
    zref: float
    outformat: OutFormat
    times: List[dt.datetime]
    download_only: bool = False
    out: Path
    verbose: bool = False
    wetFilenames: List[Path]
    hydroFilenames: List[Path]
    parallel: int = 1
    pnts_file: Path
