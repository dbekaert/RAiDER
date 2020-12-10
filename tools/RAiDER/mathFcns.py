""" General math utility functions """
import importlib
import multiprocessing as mp
import os
import re
from datetime import datetime, timedelta

import h5py
import numpy as np
import pandas as pd
import pyproj
from osgeo import gdal, osr

from RAiDER.constants import Zenith
from RAiDER import Geo2rdr
from RAiDER.logger import *

gdal.UseExceptions()


def sind(x):
    """Return the sine of x when x is in degrees."""
    return np.sin(np.radians(x))


def cosd(x):
    """Return the cosine of x when x is in degrees."""
    return np.cos(np.radians(x))


def round_time(dt, roundTo=60):
   '''
   Round a datetime object to any time lapse in seconds
   dt: datetime.datetime object
   roundTo: Closest number of seconds to round to, default 1 minute.
   Source: https://stackoverflow.com/questions/3463930/how-to-round-the-minute-of-a-datetime-object/10854034#10854034
   '''
   seconds  = (dt.replace(tzinfo=None) - dt.min).seconds
   rounding = (seconds+roundTo/2) // roundTo * roundTo
   return dt + timedelta(0,rounding-seconds,-dt.microsecond)


def round_date(date, precision):
    # First try rounding up
    # Timedelta since the beginning of time
    datedelta = datetime.min - date
    # Round that timedelta to the specified precision
    rem = datedelta % precision
    # Add back to get date rounded up
    round_up = date + rem

    # Next try rounding down
    datedelta = date - datetime.min
    rem = datedelta % precision
    round_down = date - rem

    # It's not the most efficient to calculate both and then choose, but
    # it's clear, and performance isn't critical here.
    up_diff = round_up - date
    down_diff = date - round_down

    return round_up if up_diff <= down_diff else round_down


def robmin(a):
    '''
    Get the minimum of an array, accounting for empty lists
    '''
    try:
        return np.nanmin(a)
    except ValueError:
        return 'N/A'


def robmax(a):
    '''
    Get the minimum of an array, accounting for empty lists
    '''
    try:
        return np.nanmax(a)
    except ValueError:
        return 'N/A'


def padLower(inarr):
    '''
    For an nd-array, add the lowest non-NaN data value 
    to the bottom of the array along the last axis
    '''
    new_var = _least_nonzero(inarr)
    return np.concatenate(
            (new_var[:, :, np.newaxis], inarr), 
            axis=2
        )


def _least_nonzero(a):
    """
    Return the "lowest" non-NaN value in a multi-dimensional 
    array (lowest along the last axis)

    Parameters
    ----------
    a   - numpy nd-array

    Returns
    -------
    The lowest non-NaN value in each column of a (last dim).
    The output will have a shape equal to a[...,0]

    Example
    -------
    >>> test = np.random.randn(2, 4)
    >>> test[0, 0] = np.nan
    >>> test[1, 0] = np.nan
    >>> test[1, 1] = np.nan
    >>> a = _least_nonzero(test)
    >>> print(a)
    """
    mgrid_index = tuple(slice(None, d) for d in a.shape[:-1])
    t1 = tuple(np.mgrid[mgrid_index])
    t2 = ((~np.isnan(a)).argmax(-1),)
    return a[t1 + t2]


