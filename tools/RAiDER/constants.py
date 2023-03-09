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

_g0 = np.float64(9.80665) # Standard gravitational constant
_g1 = np.float64(9.80616) # Gravitational constant @ 45° latitude used for corrections of earth's centrifugal force
_RE = np.float64(6371008.7714)

R_EARTH_MAX_WGS84 = 6378137
R_EARTH_MIN_WGS84 = 6356752

_CUBE_SPACING_IN_M = float(2000)  # Horizontal spacing of cube
_THRESHOLD_SECONDS = 1 * 60 # Threshold delta_time in seconds


