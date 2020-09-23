#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
from scipy.interpolate import interp1d

from RAiDER.interpolate import interpolate

import pandas as pd


class RegularGridInterpolator(object):
    """
    Provides a wrapper around RAiDER.interpolate.interpolate with a similar
    interface to scipy.interpolate.RegularGridInterpolator.
    """

    def __init__(
        self,
        grid,
        values,
        fill_value=None,
        assume_sorted=False,
        max_threads=8
    ):
        self.grid = grid
        self.values = values
        self.fill_value = fill_value
        self.assume_sorted = assume_sorted
        self.max_threads = max_threads

    def __call__(self, points):
        if isinstance(points, tuple):
            shape = points[0].shape
            for arr in points:
                assert arr.shape == shape, "All dimensions must contain the same number of points!"
            interp_points = np.stack(points, axis=-1)
        else:
            interp_points = points

        return interpolate(
            self.grid,
            self.values,
            interp_points,
            fill_value=self.fill_value,
            assume_sorted=self.assume_sorted,
            max_threads=self.max_threads
        )


def interp_along_axis(oldCoord, newCoord, data, axis=2, pad=False):
    '''
    DEPRECATED: Use RAiDER.interpolate.interpolate_along_axis instead (it is
    much faster). This function now primarily exists to verify the behavior of
    the new one.

    Interpolate an array of 3-D data along one axis. This function
    assumes that the x-coordinate increases monotonically.
    '''
    if oldCoord.ndim > 1:
        stackedData = np.concatenate([oldCoord, data, newCoord], axis=axis)
        out = np.apply_along_axis(interpVector, axis=axis, arr=stackedData, Nx=oldCoord.shape[axis])
    else:
        out = np.apply_along_axis(interpV, axis=axis, arr=data, old_x=oldCoord, new_x=newCoord,
                                  left=np.nan, right=np.nan)

    return out


def interpV(y, old_x, new_x, left=None, right=None, period=None):
    '''
    Rearrange np.interp's arguments
    '''
    return np.interp(new_x, old_x, y, left=left, right=right, period=period)


def interpVector(vec, Nx):
    '''
    Interpolate data from a single vector containing the original
    x, the original y, and the new x, in that order. Nx tells the
    number of original x-points.
    '''
    x = vec[:Nx]
    y = vec[Nx:2 * Nx]
    xnew = vec[2 * Nx:]
    f = interp1d(x, y, bounds_error=False, copy=False, assume_sorted=True)
    return f(xnew)


def fillna3D(array, axis=-1):
    
    XX, YY, temp2 = array.shape
    
    if ((axis == 0)|(axis == 1)):
        temp_pd = pd.DataFrame()
        temp_pd = temp_pd.interpolate(method='linear',axis=axis,limit_direction='both')
        array = temp_pd.to_numpy()
    elif (axis == -1):
        for yy in range(YY):
            temp_pd = pd.DataFrame(data=array[:,yy,:], dtype=np.float32)
            temp_pd = temp_pd.interpolate(method='linear',axis=1,limit_direction='both')
            array[:,yy,:] = temp_pd.to_numpy()
    else:
        raise Exception("Axis out of the array shape")
    
    return array
