"""Helper types used throughout RAiDER."""

from typing import Literal, TypeVar, Union

import numpy as np
from pyproj import CRS


LookDir = Literal['right', 'left']
TimeInterpolationMethod = Literal['none', 'center_time', 'azimuth_time_grid']
CRSLike = Union[CRS, str, int]


# From numpy
_ScalarT_co = TypeVar("_ScalarT_co", bound=np.generic, covariant=True)

# Helpers for type annotating the dimensions of a numpy array.
# When you use these, your type checker will alert you when an annotated array
# is used in a way inconsistent with its dimensions.
Array1D = np.ndarray[tuple[int], np.dtype[_ScalarT_co]]
Array2D = np.ndarray[tuple[int, int], np.dtype[_ScalarT_co]]
Array3D = np.ndarray[tuple[int, int, int], np.dtype[_ScalarT_co]]
# ... (repeat the pattern as needed for higher dimensions)

# Any number of dimensions -- when ndim is not able to be known ahead of time
ArrayND = np.ndarray[tuple[int, ...], np.dtype[_ScalarT_co]]


FloatArray1D = Array1D[np.floating]
FloatArray2D = Array2D[np.floating]
FloatArray3D = Array3D[np.floating]
FloatArrayND = ArrayND[np.floating]
