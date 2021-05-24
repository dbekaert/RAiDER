#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
# Copyright 2019, by the California Institute of Technology. ALL RIGHTS
# RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np

_ZMIN = np.float64(-100)   # minimum required height
_ZREF = np.float64(15000)  # maximum requierd height
_STEP = np.float64(15.0)     # integration step size in meters

_g0 = np.float64(9.80665)
_RE = np.float64(6371008.7714)

R_EARTH_MAX = 6378137
R_EARTH_MIN = 6356752

class Zenith:
    """Special value indicating a look vector of "zenith"."""
    pass


class Conventional():
    """
    Special value indicating that the zenith delay will 
    be projected using the standard cos(inc) scaling.
    """

    def __init__(self, los_file):
        raise NotImplementedError
