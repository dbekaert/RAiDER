# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from pathlib import Path
from typing import Union, float, int, bool, tuple
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from RAiDER.interpolate import interpolate


class RegularGridInterpolator:
    """
    Provides a wrapper around RAiDER.interpolate.interpolate with a similar
    interface to scipy.interpolate.RegularGridInterpolator.
    """
    def __init__(
            self,
            grid: np.ndarray,
            values: np.ndarray,
            fill_value: Union[None,float,int]=None,
            assume_sorted: bool=False,
            max_threads: int=8,
        ) -> None:
        """
        Args:
            grid (np.ndarray): _description_
            values (_type_): _description_
            fill_value (None,float, or int, optional): Fill value to represent no data
            assume_sorted (bool, optional): _description_. Defaults to False.
            max_threads (int, optional): _description_. Defaults to 8.
        """  # noqa: D202
   
        self.grid = grid
        self.values = values
        self.fill_value = fill_value
        self.assume_sorted = assume_sorted
        self.max_threads = max_threads

    def __call__(self, points: Union[tuple, np.array]) -> np.array:
        if isinstance(points, tuple):
            shape = points[0].shape
            for arr in points:
                assert arr.shape == shape, 'All dimensions must contain the same number of points!'
            interp_points = np.stack(points, axis=-1)
            in_shape = interp_points.shape
        elif points.ndim > 2:
            in_shape = points.shape
            interp_points = points.reshape((np.prod(points.shape[:-1]),) + (points.shape[-1],))
        else:
            interp_points = points
            in_shape = interp_points.shape

        out = interpolate(
            self.grid,
            self.values,
            interp_points,
            fill_value=self.fill_value,
            assume_sorted=self.assume_sorted,
            max_threads=self.max_threads,
        )
        return out.reshape(in_shape[:-1])


def interp_along_axis(oldCoord, newCoord, data, axis=2, pad=False):
    """
    DEPRECATED: Use RAiDER.interpolate.interpolate_along_axis instead (it is
    much faster). This function now primarily exists to verify the behavior of
    the new one.

    Interpolate an array of 3-D data along one axis. This function
    assumes that the x-coordinate increases monotonically.
    """
    if oldCoord.ndim > 1:
        stackedData = np.concatenate([oldCoord, data, newCoord], axis=axis)
        out = np.apply_along_axis(interpVector, axis=axis, arr=stackedData, Nx=oldCoord.shape[axis])
    else:
        out = np.apply_along_axis(
            interpV, axis=axis, arr=data, old_x=oldCoord, new_x=newCoord, left=np.nan, right=np.nan
        )

    return out


def interpV(y, old_x, new_x, left=None, right=None, period=None):
    """Rearrange np.interp's arguments."""
    return np.interp(new_x, old_x, y, left=left, right=right, period=period)


def interpVector(vec, Nx):
    """
    Interpolate data from a single vector containing the original
    x, the original y, and the new x, in that order. Nx tells the
    number of original x-points.
    """
    x = vec[:Nx]
    y = vec[Nx : 2 * Nx]
    xnew = vec[2 * Nx :]
    f = interp1d(x, y, bounds_error=False, copy=False, assume_sorted=True)
    return f(xnew)


def fillna3D(array, axis=-1, fill_value=0.0):
    """
    This function fills in NaNs in 3D arrays, specifically using the nearest non-nan value
    for "low" NaNs and 0s for "high" NaNs.

    Arguments:
        array   - 3D array, where the last axis is the "z" dimension

    Returns:
        3D array with low NaNs filled as nearest neighbors and high NaNs filled as 0s
    """
    # fill lower NaNs with nearest neighbor
    narr = np.moveaxis(array, axis, -1)
    nars = narr.reshape((np.prod(narr.shape[:-1]),) + (narr.shape[-1],))
    dfd = pd.DataFrame(data=nars).interpolate(axis=1, limit_direction='backward')
    out = dfd.values.reshape(array.shape)

    # fill upper NaNs with 0s
    outmat = np.moveaxis(out, -1, axis)
    outmat[np.isnan(outmat)] = fill_value
    return outmat


def interpolateDEM(dem_path: Union[Path, str], outLL: tuple[np.ndarray, np.ndarray], method='nearest') -> np.ndarray:
    """Interpolate a DEM raster to a set of lat/lon query points using rioxarray.

    outLL will be a tuple of (lats, lons). lats/lons can either be 1D arrays or 2
        For now will only use first row/col of 2D
    """
    import rioxarray as xrr
    from xarray import Dataset

    data = xrr.open_rasterio(dem_path, band_as_variable=True)
    assert isinstance(data, Dataset), 'DEM could not be opened as a rioxarray dataset'
    da_dem = data['band_1']
    lats, lons = outLL
    lats = lats[:, 0] if lats.ndim == 2 else lats
    lons = lons[0, :] if lons.ndim == 2 else lons
    z_out: np.ndarray = da_dem.interp(y=np.sort(lats)[::-1], x=lons).data
    return z_out
