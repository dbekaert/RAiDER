#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Brett Buzzanga, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import datetime
import os.path
import shelve
import xml.etree.ElementTree as ET

import numpy as np

import RAiDER.utilFcns as utilFcns
from RAiDER import Geo2rdr
from RAiDER.constants import _ZREF, Zenith


_SLANT_RANGE_THRESH = 5e6


def getLookVectors(look_vecs, lats, lons, heights, time=None,  pad=3*3600):
    '''
    Get unit look vectors pointing from the ground (target) pixels to the sensor,
    or to Zenith. Can be accomplished using an ISCE-style 2-band LOS file or a
    file containing orbital statevectors.

    *NOTE*:
    These line-of-sight vectors will NOT match ordinary LOS vectors for InSAR
    because they are in an ECEF reference frame instead of a local ENU. This is done
    because the construction of rays is done in ECEF rather than the local ENU.

    Parameters
    ----------
    look_vecs: LookVector object or tuple  - Either a Zenith object or a tuple,
                                             with the second element containing
                                             the name of either a line-of-sight
                                             file or orbital statevectors file
    lats/lons/heights: ndarray             - WGS-84 coordinates of the target pixels
    time: python datetime                  - user-requested query time. Must be
                                             compatible with the orbit file passed.
                                             Only required for a statevector file.
    pad: int                               - integer number of seconds to pad around
                                             the user-specified time; default 3 hours
                                             Only required for a statevector file.

    Returns
    -------
    look_vecs: ndarray  - an <in_shape> x 3 array of unit look vectors, defined in an
                          Earth-centered, earth-fixed reference frame (ECEF). Convention is
                          vectors point from the target pixel to the sensor.

    Example:
    --------
    >>> from RAiDER.constants import Zenith
    >>> from RAiDER.losreader import getLookVectors
    >>> import numpy as np
    >>> getLookVectors(Zenith, np.array([0]), np.array([0]), np.array([0]))
    >>> # array([[1, 0, 0]])
    '''
    if (look_vecs is None) or (look_vecs is Zenith):
        look_vecs = Zenith
    else:
        look_vecs = look_vecs[1]

    in_shape = lats.shape

    if look_vecs is Zenith:
        look_vecs = getZenithLookVecs(lats, lons, heights)
    else:
        try:
            LOS_enu = inc_hd_to_enu(*utilFcns.gdal_open(look_vecs))
            look_vecs = utilFcns.enu2ecef(
                    LOS_enu[...,0],
                    LOS_enu[...,1],
                    LOS_enu[...,2],
                    lats,
                    lons,
                    heights
                )

        # if that doesn't work, try parsing as a statevector (orbit) file
        except OSError:
            svs       = get_sv(look_vecs, time, pad)
            look_vecs = state_to_los(*svs, lats=lats, lons=lons, heights=heights)

        # Otherwise, throw an error
        except:
            raise ValueError(
                'getLookVectors: I cannot parse the file {}'.format(look_vecs)
            )

    mask = (np.isnan(heights) | np.isnan(lats) | np.isnan(lons))
    look_vecs[mask, :] = np.nan

    return look_vecs.astype(np.float64)


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


def get_sv(los_file, ref_time, pad=3*3600):
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
            svs = read_ESA_Orbit_file(los_file, ref_time)
        except:
            try:
                svs = read_shelve(los_file)
            except:
                raise ValueError(
                    'get_sv: I cannot parse the statevector file {}'.format(los_file)
                )

    idx = cut_times(svs[0], pad=pad)
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

    east = utilFcns.sind(incidence) * utilFcns.cosd(heading + 90)
    north = utilFcns.sind(incidence) * utilFcns.sind(heading + 90)
    up = utilFcns.cosd(incidence)

    return np.stack((east, north, up), axis=-1)


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
    '''
    Converts information from a state vector for a satellite orbit, given in terms of
    position and velocity, to line-of-sight information at each (lon,lat, height)
    coordinate requested by the user.

    Parameters
    ----------
    t, x, y, z, vx, vy, vz  	- time, position, and velocity in ECEF of the sensor
    lats, lons, heights     	- Ellipsoidal (WGS84) positions of target ground pixels

    Returns
    -------
    LOS 			- * x 3 matrix of LOS unit vectors in ECEF (*not* ENU)

    Example:
    >>> import datetime
    >>> from RAiDER.utilFcns import gdal_open
    >>> import RAiDER.losreader as losr
    >>> lats, lons, heights = np.array([-76.1]), np.array([36.83]), np.array([0])
    >>> time = datetime.datetime(2018,11,12,23,0,0)
    >>> # download the orbit file beforehand
    >>> esa_orbit_file = 'S1A_OPER_AUX_POEORB_OPOD_20181203T120749_V20181112T225942_20181114T005942.EOF'
    >>> svs = losr.read_ESA_Orbit_file(esa_orbit_file, time)
    >>> LOS = losr.state_to_los(*svs, lats=lats, lons=lons, heights=heights)
    '''

    # check the inputs
    if t.size < 4:
        raise RuntimeError(
            'state_to_los: At least 4 state vectors are required'
            ' for orbit interpolation'
        )
    if t.shape != x.shape:
        raise RuntimeError('state_to_los: t and x must be the same size')
    if lats.shape != lons.shape:
        raise RuntimeError('state_to_los: lats and lons must be the same size')

    in_shape = lats.shape
    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)

    los_x, los_y, los_z = [], [], []
    for i, (l, L, h) in enumerate(zip(lats.ravel(), lons.ravel(), heights.ravel())):
        # Geo2rdr is picky about how the heights look
        height_array = np.array(((h,),)).astype(np.double)

        # Set the target pixel location
        geo2rdr_obj.set_geo_coordinate(np.radians(L), np.radians(l), 1, 1, height_array)

        # compute the look vector and target-sensor range in ECEF
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        # LOS is defined as pointing from the ground pixel to the sensor
        los = np.squeeze(geo2rdr_obj.get_los())
        los_x.append(los[0])
        los_y.append(los[1])
        los_z.append(los[2])

    los_ecef = np.stack([
        np.array(los_x).reshape(in_shape),
        np.array(los_y).reshape(in_shape),
        np.array(los_z).reshape(in_shape),
    ], axis=-1)

    # Sanity check for purpose of tracking problems
    # print ('Slant range', geo2rdr_obj.get_slant_range())
    if geo2rdr_obj.get_slant_range() > _SLANT_RANGE_THRESH:
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
    del geo2rdr_obj

    return los_ecef


def cut_times(times, pad=3600*3):
    """
    Slice the orbit file around the reference aquisition time. This is done
    by default using a three-hour window, which for Sentinel-1 empirically
    works out to be roughly the largest window allowed by the orbit time.

    Parameters
    ----------
    times: Nt x 1 ndarray     - Vector of orbit times as seconds since the
                                user-requested time
    pad: int                  - integer time in seconds to use as padding

    Returns
    -------
    idx: Nt x 1 logical ndarray - a mask of times within the padded request time.
    """
    return np.abs(times) < pad


def read_shelve(filename):
    #TODO: docstring and unit tests
    with shelve.open(filename, 'r') as db:
        obj = db['frame']

    numSV = len(obj.orbit.stateVectors)

    t = np.ones(numSV)
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t[i] = st.time.second + st.time.minute * 60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

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
                t_, x_, y_, z_, vx_, vy_, vz_ = [float(t) for t in line.split()]
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
    return [np.array(a) for a in [t, x, y, z, vx, vy, vz]]


def read_ESA_Orbit_file(filename, ref_time):
    '''
    Read orbit data from an orbit file supplied by ESA

    Parameters
    ----------
    filename: str             - string of the orbit filename
    ref_time: python datetime - user requested python datetime

    Returns
    -------
    t: Nt x 1 ndarray   - a numpy vector with Nt elements containing time
                          in seconds since the reference time, within "pad"
                          seconds of the reference time
    x, y, z: Nt x 1 ndarrays    - x/y/z positions of the sensor at the times t
    vx, vy, vz: Nt x 1 ndarrays - x/y/z velocities of the sensor at the times t
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    data_block = root[1]
    numOSV = len(data_block[0])

    t = np.ones(numOSV)
    x = np.ones(numOSV)
    y = np.ones(numOSV)
    z = np.ones(numOSV)
    vx = np.ones(numOSV)
    vy = np.ones(numOSV)
    vz = np.ones(numOSV)

    times = []
    for i, st in enumerate(data_block[0]):
        t[i] = (
            datetime.datetime.strptime(
                st[1].text,
                'UTC=%Y-%m-%dT%H:%M:%S.%f'
            ) - ref_time
        ).total_seconds()

        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i] = float(st[7].text)
        vy[i] = float(st[8].text)
        vz[i] = float(st[9].text)

    return [t, x, y, z, vx, vy, vz]
