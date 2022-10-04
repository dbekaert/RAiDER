#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Brett Buzzanga, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import datetime
import shelve

import xml.etree.ElementTree as ET
import numpy as np
from abc import ABC
from scipy.interpolate import interp1d

import isce3.ext.isce3 as isce3

from RAiDER.utilFcns import (
    cosd, sind, gdal_open, enu2ecef, lla2ecef, ecef2enu
)
from RAiDER.constants import _ZREF


_SLANT_RANGE_THRESH = 5e6


class LOS(ABC):
    '''LOS Class definition for handling look vectors'''

    def __init__(self):
        self._lats, self._lons, self._heights = None, None, None
        self._look_vecs = None

    def setPoints(self, lats, lons=None, heights=None):
        '''Set the pixel locations'''
        if (lats is None) and (self._lats is None):
            raise RuntimeError("You haven't given any point locations yet")

        # Will overwrite points by default
        if lons is None:
            llh = lats  # assume points are [lats lons heights]
            self._lats = llh[..., 0]
            self._lons = llh[..., 1]
            self._heights = llh[..., 2]
        elif heights is None:
            self._lats = lats
            self._lons = lons
            self._heights = np.zeros((len(lats), 1))
        else:
            self._lats = lats
            self._lons = lons
            self._heights = heights

    def getXYZ(self):
        self._xyz = lla2ecef(self._lats, self._lons, self._heights)


class Zenith(LOS):
    """Special value indicating a look vector of "zenith"."""

    def setLookVectors(self):
        '''Set point locations and calculate Zenith look vectors'''
        if self._lats is None:
            raise ValueError('Target points not set')
        if self._look_vecs is None:
            self._look_vecs = getZenithLookVecs(self._lats, self._lons, self._heights)

    def __call__(self, delays):
        '''Placeholder method for consistency with the other classes'''
        return delays


class Conventional(LOS):
    """
    Special value indicating that the zenith delay will
    be projected using the standard cos(inc) scaling.
    """
    def __init__(self, filename=None, time=None, pad=None):
        super().__init__()
        self._file = filename
        self._time = time
        self._pad = pad

    def __call__(self, delays):
        '''Read the LOS file and convert it to look vectors'''
        if self._lats is None:
            raise ValueError('Target points not set')
        if self._file is None:
            raise ValueError('LOS file not set')

        try:
            # if an ISCE-style los file is passed open it with GDAL
            LOS_enu = inc_hd_to_enu(*gdal_open(self._filename))
        except:
            # Otherwise, treat it as an orbit / statevector file
            svs = np.stack(get_sv(self._filename, self._time, self._pad), axis=-1)
            self.getXYZ()
            LOS_enu = state_to_los(svs, self._xyz)

        return delays / LOS_enu[...,-1]


class Raytracing(LOS):
    """
    Special value indicating that full raytracing will be
    used to calculate slant delays.

    Get unit look vectors pointing from the ground (target) pixels to the sensor,
    or to Zenith. Can be accomplished using an ISCE-style 2-band LOS file or a
    file containing orbital statevectors.

    *NOTE*:
    These line-of-sight vectors will NOT match ordinary LOS vectors for InSAR
    because they are in an ECEF reference frame instead of a local ENU. This is done
    because the construction of rays is done in ECEF rather than the local ENU.

    Parameters
    ----------
    time: python datetime                  - user-requested query time. Must be
                                             compatible with the orbit file passed.
                                             Only required for a statevector file.
    pad: int                               - integer number of seconds to pad around
                                             the user-specified time; default 3 hours
                                             Only required for a statevector file.

    Returns
    -------
    look_vecs: ndarray  - an <in_shape> x 3 array of unit look vectors, defined in
                          an Earth-centered, earth-fixed reference frame (ECEF).
                          Convention is vectors point from the target pixel to the
                          sensor.
    lengths: ndarray    - array of <in_shape> of the distnce from the surface to
                          the top of the troposphere (denoted by zref)

    Example:
    --------
    >>> from RAiDER.losreader import Raytracing
    >>> import numpy as np

>>> #TODO
    >>> # 
    """
    def __init__(self, filename=None):
        '''read in and parse a statevector file'''
        self._file = filename

    def getLookVectors(self, time, pad=3*60):
        '''
        Calculate look vectors for raytracing
        '''
        svs = np.stack(get_sv(self._file, time, pad), axis=-1)
        self.getXYZ()
        LOS_enu = state_to_los(svs, self._xyz)
        self._look_vecs = enu2ecef(
            LOS_enu[..., 0],
            LOS_enu[..., 1],
            LOS_enu[..., 2],
            self._lats,
            self._lons,
            self._heights
        )

    def calculateDelays(self, delays):
        '''
        Here "delays" is point-wise delays (i.e. refractivities), not 
        integrated ZTD/STD.
        '''
        # Create rays
        # Interpolate delays to rays
        # Integrate along rays
        # Return STD 
        raise NotImplementedError


def getZenithLookVecs(lats, lons, heights):
    '''
    Returns look vectors when Zenith is used.

    Parameters
    ----------
    lats/lons/heights: ndarray  - Numpy arrays containing WGS-84 target locations

    Returns
    -------
    zenLookVecs: ndarray         - (in_shape) x 3 unit look vectors in an ECEF reference frame
    '''
    x = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    y = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    z = np.sin(np.radians(lats))

    return np.stack([x, y, z], axis=-1)


def get_sv(los_file, ref_time, pad=3*60):
    """
    Read an LOS file and return orbital state vectors

    Parameters
    ----------
    los_file: str             - user-passed file containing either look
                                vectors or statevectors for the sensor
    ref_time: python datetime - User-requested datetime; if not encompassed
                                by the orbit times will raise a ValueError
    pad: int                  - number of seconds to keep around the
                                requested time

    Returns
    -------
    svs: 7 x 1 list of Nt x 1 ndarrays - the times, x/y/z positions and
                                         velocities of the sensor for the given
                                         window around the reference time
    """
    try:
        svs = read_txt_file(los_file)
    except ValueError:
        try:
            svs = read_ESA_Orbit_file(los_file)
        except BaseException:
            try:
                svs = read_shelve(los_file)
            except BaseException:
                raise ValueError(
                    'get_sv: I cannot parse the statevector file {}'.format(los_file)
                )

    idx = cut_times(svs[0], ref_time, pad=pad)
    svs = [d[idx] for d in svs]
    return svs


def inc_hd_to_enu(incidence, heading):
    '''
    Convert incidence and heading to line-of-sight vectors from the ground to the top of
    the troposphere.

    Parameters
    ----------
    incidence: ndarray	       - incidence angle in deg from vertical
    heading: ndarray 	       - heading angle in deg clockwise from north
    lats/lons/heights: ndarray - WGS84 ellipsoidal target (ground pixel) locations

    Returns
    -------
    LOS: ndarray  - (input_shape) x 3 array of unit look vectors in local ENU

    Algorithm referenced from http://earthdef.caltech.edu/boards/4/topics/327
    '''
    if np.any(incidence < 0):
        raise ValueError('inc_hd_to_enu: Incidence angle cannot be less than 0')

    east = sind(incidence) * cosd(heading + 90)
    north = sind(incidence) * sind(heading + 90)
    up = cosd(incidence)

    return np.stack((east, north, up), axis=-1)


def read_shelve(filename):
    # TODO: docstring and unit tests
    with shelve.open(filename, 'r') as db:
        obj = db['frame']

    numSV = len(obj.orbit.stateVectors)
    if numSV == 0:
        raise ValueError('read_shelve: the file has not statevectors')

    t = []
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t.append(st.time)
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    t = np.array(t)
    return t, x, y, z, vx, vy, vz


def read_txt_file(filename):
    '''
    Read a 7-column text file containing orbit statevectors. Time
    should be denoted as integer time in seconds since the reference
    epoch (user-requested time).

    Parameters
    ----------
    filename: str  - user-supplied space-delimited text file with no header
                     containing orbital statevectors as 7 columns:
                     - time in seconds since the user-supplied epoch
                     - x / y / z locations in ECEF cartesian coordinates
                     - vx / vy / vz velocities in m/s in ECEF coordinates
    Returns
    svs: list      - a length-7 list of numpy vectors containing the above
                     variables
    '''
    t = list()
    x = list()
    y = list()
    z = list()
    vx = list()
    vy = list()
    vz = list()
    with open(filename, 'r') as f:
        for line in f:
            try:
                parts = line.strip().split()
                t_ = datetime.datetime.fromisoformat(parts[0])
                x_, y_, z_, vx_, vy_, vz_ = [float(t) for t in parts[1:]]
            except ValueError:
                raise ValueError(
                    "I need {} to be a 7 column text file, with ".format(filename) +
                    "columns t, x, y, z, vx, vy, vz (Couldn't parse line " +
                    "{})".format(repr(line)))
            t.append(t_)
            x.append(x_)
            y.append(y_)
            z.append(z_)
            vx.append(vx_)
            vy.append(vy_)
            vz.append(vz_)

    if len(t) < 4:
        raise ValueError('read_txt_file: File {} does not have enough statevectors'.format(filename))

    return [np.array(a) for a in [t, x, y, z, vx, vy, vz]]


def read_ESA_Orbit_file(filename):
    '''
    Read orbit data from an orbit file supplied by ESA

    Parameters
    ----------
    filename: str             - string of the orbit filename

    Returns
    -------
    t: Nt x 1 ndarray   - a numpy vector with Nt elements containing time
                          in python datetime
    x, y, z: Nt x 1 ndarrays    - x/y/z positions of the sensor at the times t
    vx, vy, vz: Nt x 1 ndarrays - x/y/z velocities of the sensor at the times t
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    data_block = root[1]
    numOSV = len(data_block[0])

    t = []
    x = np.ones(numOSV)
    y = np.ones(numOSV)
    z = np.ones(numOSV)
    vx = np.ones(numOSV)
    vy = np.ones(numOSV)
    vz = np.ones(numOSV)

    for i, st in enumerate(data_block[0]):
        t.append (
            datetime.datetime.strptime(
                st[1].text,
                'UTC=%Y-%m-%dT%H:%M:%S.%f'
            )
        )

        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i] = float(st[7].text)
        vy[i] = float(st[8].text)
        vz[i] = float(st[9].text)
    t = np.array(t)
    return [t, x, y, z, vx, vy, vz]


############################
def state_to_los(svs, xyz_targets):
    '''
    Converts information from a state vector for a satellite orbit, given in terms of
    position and velocity, to line-of-sight information at each (lon,lat, height)
    coordinate requested by the user.
    Parameters
    ----------
    svs            - t, x, y, z, vx, vy, vz - time, position, and velocity in ECEF of the sensor
    xyz_targets    - lats, lons, heights - Ellipsoidal (WGS84) positions of target ground pixels
    Returns
    -------
    LOS 			- * x 3 matrix of LOS unit vectors in ECEF (*not* ENU)
    Example:
    >>> import datetime
    >>> import numpy
    >>> from RAiDER.utilFcns import gdal_open
    >>> import RAiDER.losreader as losr
    >>> lats, lons, heights = np.array([-76.1]), np.array([36.83]), np.array([0])
    >>> time = datetime.datetime(2018,11,12,23,0,0)
    >>> # download the orbit file beforehand
    >>> esa_orbit_file = 'S1A_OPER_AUX_POEORB_OPOD_20181203T120749_V20181112T225942_20181114T005942.EOF'
    >>> svs = losr.read_ESA_Orbit_file(esa_orbit_file)
    >>> LOS = losr.state_to_los(*svs, lats=lats, lons=lons, heights=heights)
    '''

    # check the inputs
    if np.min(svs.shape) < 4:
        raise RuntimeError(
            'state_to_los: At least 4 state vectors are required'
            ' for orbit interpolation'
        )

    # Flatten the input array for convenience
    in_shape = xyz_targets.shape
    target_xyz = np.stack([xyz_targets[..., 0].flatten(), xyz_targets[..., 1].flatten(), xyz_targets[..., 2].flatten()], axis=-1)
    Npts = len(target_xyz)

    # Iterate through targets and compute LOS
    slant_range = []
    los = np.empty((Npts, 3), dtype=np.float64)
    for k in range(Npts):
        if ~any(np.isnan(target_xyz[k,:])):
            los[k, :], sr = get_radar_coordinate(target_xyz[k, :], svs)
            slant_range.append(sr)
        else:
            slant_range.append(np.nan)
    slant_ranges = np.array(slant_range)

    # Sanity check for purpose of tracking problems
    if slant_ranges.max() > _SLANT_RANGE_THRESH:
        raise RuntimeError(
            '''
            state_to_los:
            It appears that your input datetime and/or orbit file does not
            correspond to the lats/lons that you've passed. Please verify
            that the input datetime is the closest possible to the
            acquisition times of the interferogram, and the orbit file covers
            the same range of time.
            '''
        )

    los_ecef = los.reshape(in_shape)
    return los_ecef


def cut_times(times, ref_time, pad=3600 * 3):
    """
    Slice the orbit file around the reference aquisition time. This is done
    by default using a three-hour window, which for Sentinel-1 empirically
    works out to be roughly the largest window allowed by the orbit time.
    Parameters
    ----------
    times: Nt x 1 ndarray     - Vector of orbit times as datetime
    ref_time: datetime        - Reference time
    pad: int                  - integer time in seconds to use as padding
    Returns
    -------
    idx: Nt x 1 logical ndarray - a mask of times within the padded request time.
    """
    diff = np.array(
        [(x-ref_time).total_seconds() for x in times]
    )
    return np.abs(diff) < pad


def get_radar_coordinate(xyz, svs, t0=None):
    '''
    Calculate the coordinate of the sensor in ECEF at the time corresponding to ***.

    Parameters
    ----------
    svs: ndarray   - Nt x 7 matrix of statevectors: [t x y z vx vy vz]
    xyz: ndarray   - position of the target in ECEF
    t0: double     - starting point of the time at which the sensor imaged the target xyz

    Returns
    -------
    sensor_xyz: ndarray  - position of the sensor in ECEF
    '''
    # initialize search
    if t0 is None:
        t = np.nanmean(svs[:, 0])
    else:
        t = t0

    dt = 1.0
    num_iteration = 20
    residual_threshold = 0.000000001

    dts = []
    for k in range(num_iteration):
        x = interpolate(svs[:, 0], svs[:, 1], t)
        y = interpolate(svs[:, 0], svs[:, 2], t)
        z = interpolate(svs[:, 0], svs[:, 3], t)
        vx = interpolate(svs[:, 0], svs[:, 4], t)
        vy = interpolate(svs[:, 0], svs[:, 5], t)
        vz = interpolate(svs[:, 0], svs[:, 6], t)
        E1 = vx * (xyz[0] - x) + vy * (xyz[1] - y) + vz * (xyz[2] - z)
        dE1 = vx * vx + vy * vy + vz * vz
        dt = E1 / dE1
        dts.append(dt)
        t = t + dt
        if np.abs(dt) < residual_threshold:
            break

    los_x = xyz[0] - x
    los_y = xyz[1] - y
    los_z = xyz[2] - z

    slant_range = np.sqrt(
        np.square(los_x) + np.square(los_y) + np.square(los_z)
    )
    return np.array([los_x, los_y, los_z]) / slant_range, slant_range


def interpolate(t, var, tq):
    '''
    Interpolate a set of statevectors to the requested input time

    Parameters
    ----------
    statevectors: ndarray   - an Nt x 7 matrix of statevectors: [t x y z vx vy vz]
    tref: double            - reference time requested (must be in the scope of t)

    Returns
    -------
    x, y, z: double    - sensor position in ECEF
    vx, vy, vz: double - sensor velocity
    '''
    f = interp1d(t, var)
    return f(tq)
