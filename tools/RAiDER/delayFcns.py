# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import multiprocessing as mp
import xarray

import numpy as np
from scipy.interpolate import RegularGridInterpolator as Interpolator


def getInterpolators(wm_file, kind='pointwise', shared=False):
    '''
    Read 3D gridded data from a processed weather model file and wrap it with
    the scipy RegularGridInterpolator
    '''
    # Get the weather model data
    try:
        ds = xarray.load_dataset(wm_file)
    except ValueError:
        ds = wm_file

    xs_wm = np.array(ds.variables['x'][:])
    ys_wm = np.array(ds.variables['y'][:])
    zs_wm = np.array(ds.variables['z'][:])
    wet = ds.variables['wet_total' if kind=='total' else 'wet'][:]
    hydro = ds.variables['hydro_total' if kind=='total' else 'hydro'][:]

    wet = np.array(wet).transpose(1, 2, 0)
    hydro = np.array(hydro).transpose(1, 2, 0)

    if np.any(np.isnan(wet)) or np.any(np.isnan(hydro)):
        raise RuntimeError(f'Weather model {wm_file} contains NaNs')

    # If shared interpolators are requested
    # The arrays are not modified - so turning off lock for performance
    if shared:
        xs_wm = make_shared_raw(xs_wm)
        ys_wm = make_shared_raw(ys_wm)
        zs_wm = make_shared_raw(zs_wm)
        wet = make_shared_raw(wet)
        hydro = make_shared_raw(hydro)


    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan, bounds_error = False)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan, bounds_error = False)

    return ifWet, ifHydro


def make_shared_raw(inarr):
    """
    Make numpy view array of mp.Array
    """
    # Create flat shared array
    shared_arr = mp.RawArray('d', inarr.size)
    # Create a numpy view of it
    shared_arr_np = np.ndarray(inarr.shape, dtype=np.float64,
                               buffer=shared_arr)
    # Copy data to shared array
    np.copyto(shared_arr_np, inarr)

    return shared_arr_np


def interpolate2(fun, x, y, z):
    '''
    helper function to make the interpolation step cleaner
    '''
    in_shape = x.shape
    out = fun((y.ravel(), x.ravel(), z.ravel()))  # note that this re-ordering is on purpose to match the weather model
    outData = out.reshape(in_shape)
    return outData
