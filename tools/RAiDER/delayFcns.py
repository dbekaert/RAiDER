# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson, David Bekaert & Yang Lei
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    import multiprocessing as mp
except ImportError:
    mp = None

from pathlib import Path
from typing import Union

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator as Interpolator

from RAiDER.logger import logger


# TODO(garlic-os): type annotate the choices for kind
def getInterpolators(wm_file: Union[xr.Dataset, Path, str], kind: str='pointwise', shared: bool=False) -> tuple[Interpolator, Interpolator]:
    """
    Read 3D gridded data from a processed weather model file and wrap it with
    the scipy RegularGridInterpolator.

    The interpolator grid is (y, x, z)
    """
    # Get the weather model data
    ds = wm_file if isinstance(wm_file, xr.Dataset) else xr.load_dataset(wm_file)

    xs_wm = np.array(ds.variables['x'][:])
    ys_wm = np.array(ds.variables['y'][:])
    zs_wm = np.array(ds.variables['z'][:])

    wet = ds.variables['wet_total' if kind == 'total' else 'wet'][:]
    hydro = ds.variables['hydro_total' if kind == 'total' else 'hydro'][:]

    wet = np.array(wet).transpose(1, 2, 0)
    hydro = np.array(hydro).transpose(1, 2, 0)

    if np.any(np.isnan(wet)) or np.any(np.isnan(hydro)):
        logger.critical('Weather model contains NaNs!')

    # If shared interpolators are requested
    # The arrays are not modified - so turning off lock for performance
    if shared:
        xs_wm = make_shared_raw(xs_wm)
        ys_wm = make_shared_raw(ys_wm)
        zs_wm = make_shared_raw(zs_wm)
        wet = make_shared_raw(wet)
        hydro = make_shared_raw(hydro)

    ifWet = Interpolator((ys_wm, xs_wm, zs_wm), wet, fill_value=np.nan, bounds_error=False)
    ifHydro = Interpolator((ys_wm, xs_wm, zs_wm), hydro, fill_value=np.nan, bounds_error=False)

    return ifWet, ifHydro


def make_shared_raw(inarr):
    """Make numpy view array of mp.Array."""
    # Create flat shared array
    if mp is None:
        raise ImportError('multiprocessing is not available')

    shared_arr = mp.RawArray('d', inarr.size)
    # Create a numpy view of it
    shared_arr_np = np.ndarray(inarr.shape, dtype=np.float64, buffer=shared_arr)
    # Copy data to shared array
    np.copyto(shared_arr_np, inarr)

    return shared_arr_np
