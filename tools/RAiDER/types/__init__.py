"""Types specific to RAiDER."""

from typing import Literal, Union

from pyproj import CRS


LookDir = Literal['right', 'left']
TimeInterpolationMethod = Literal['none', 'center_time', 'azimuth_time_grid']
CRSLike = Union[CRS, str, int]
