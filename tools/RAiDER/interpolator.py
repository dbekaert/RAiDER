#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np

from RAiDER.utilFcns import parallel_apply_along_axis


def interp_along_axis(oldCoord, newCoord, data, axis=2, pad=False):
    '''
    Interpolate an array of 3-D data along one axis. This function
    assumes that the x-coordinate increases monotonically.
    '''
    if oldCoord.ndim > 1:
        stackedData = np.concatenate([oldCoord, data, newCoord], axis=axis)
        try:
            out = parallel_apply_along_axis(interpVector, arr=stackedData, axis=axis, Nx=oldCoord.shape[axis])
        except:
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
    from scipy.interpolate import interp1d
    x = vec[:Nx]
    y = vec[Nx:2*Nx]
    xnew = vec[2*Nx:]
    f = interp1d(x, y, bounds_error=False)
    return f(xnew)


def _interp3D(xs, ys, zs, values, zlevels, shape=None):
    '''
    3-D interpolation on a non-uniform grid, where z is non-uniform but x, y are uniform
    '''
    from scipy.interpolate import RegularGridInterpolator as rgi

    interp = rgi((ys, xs, zs), values, bounds_error=False, fill_value=np.nan)
    return interp


def fillna3D(array, axis=-1):
    '''
    Fcn to fill in NaNs in a 3D array by interpolating over one axis only
    '''
    # Need to handle each axis
    narr = np.moveaxis(array, axis, -1)
    shape = narr.shape
    y = narr.flatten()

    test_nan = np.isnan(y)
    finder = lambda z: z.nonzero()[0]

    try:
        y[test_nan] = np.interp(finder(test_nan), finder(~test_nan), y[~test_nan])
        newy = np.reshape(y, shape)
        final = np.moveaxis(newy, -1, axis)
        return final
    except ValueError:
        return array
