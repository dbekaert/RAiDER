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


def getLookVectors(look_vecs, lats, lons, heights, time,  pad=3*3600):
    '''
    If the input look vectors are specified as Zenith, compute and return the
    look vectors. Otherwise, check that the look_vecs shape makes sense.
    '''
    if (look_vecs is None) or (look_vecs is Zenith):
        look_vecs = Zenith
    else:
        look_vecs = look_vecs[1]

    in_shape = lats.shape
    lat = lats.flatten()
    lon = lons.flatten()
    hgt = heights.flatten()

    if look_vecs is Zenith:
        look_vecs = _getZenithLookVecs(lat, lon, hgt)
    else:
        look_vecs = infer_los(look_vecs, lat, lon, hgt, time, pad)

    mask = np.isnan(hgt) | np.isnan(lat) | np.isnan(lon)
    look_vecs[mask, :] = np.nan

    return look_vecs.reshape(in_shape + (3,)).astype(np.float64)


def _getZenithLookVecs(lats, lons, heights):
    '''
    Returns look vectors when Zenith is used.

    Parameters
    ----------
    lats/lons/heights - Nx1 numpy arrays of points.

    Returns
    -------
    zenLookVecs       - an Nx3 numpy array with the unit look vectors in an ECEF
                        reference frame.
    '''
    try:
        if (lats.ndim != 1) | (heights.ndim != 1) | (lons.ndim != 1):
            raise ValueError(
                '_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays'
            )
    except AttributeError:
        raise ValueError(
            '_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays'
        )

    e = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    n = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    u = np.sin(np.radians(lats))

    ecef = utilFcns.enu2ecef(e, n, u, lats, lons, heights).T
    return ecef.astype(np.float64)


def infer_los(los_file, lats, lons, heights, time, pad=3*3600):
    '''
    Helper function to deal with various LOS files supplied
    '''
    # Assume that the user passed a line-of-sight file
    try:
        incidence, heading = [f.flatten() for f in utilFcns.gdal_open(los_file)]
        utilFcns.checkShapes(
                np.stack(
                    (incidence, heading), 
                    axis=-1
                ), 
                lats, 
                lons, 
                heights
            )
        LOS_enu = los_to_lv(incidence, heading, lats, lons, heights)
        LOS = utilFcns.enu2ecef(
                LOS_enu[...,0], 
                LOS_enu[...,1], 
                LOS_enu[...,2], 
                lats, 
                lons, 
                heights
            )

    # if that doesn't work, try parsing as a statevector (orbit) file
    except OSError:
        svs = get_sv(los_file, time, pad)
        LOS = state_to_los(*svs, lats=lats, lons=lons, heights=heights)

    # Otherwise, throw an error
    except:
        raise ValueError('infer_los: I cannot parse the file {}'.format(los_file))

    return LOS


def get_sv(los_file, ref_time, pad=3*3600):
    """
    Read an LOS file.

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


def los_to_lv(incidence, heading, lats, lons, heights):
    '''
    Convert incidence and heading to line-of-sight vectors from the ground to the top of
    the troposphere.

    Parameters
    ----------
    incidence		- Numpy array containing incidence angle (deg from vertical)
    heading 		- Numpy array containing heading angle (deg clockwise from north) #TODO: check this
    lats, lons, heights - Numpy arrays with WGS84 ellipsoidal target (ground pixel) locations 

    Returns
    -------
    LOS  		- * x 3 matrix of unit look vectors in local ENU reference frame

    Algorithm referenced from http://earthdef.caltech.edu/boards/4/topics/327
    '''
    a_0 = incidence
    a_1 = heading

    east = utilFcns.sind(a_0) * utilFcns.cosd(a_1 + 90)
    north = utilFcns.sind(a_0) * utilFcns.sind(a_1 + 90)
    up = utilFcns.cosd(a_0)
    
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

    *NOTE*:
    These line-of-sight vectors will NOT match ordinary LOS vectors for InSAR 
    because they are in an ECEF reference frame instead of a local ENU. This is done 
    because the construction of rays is done in ECEF rather than the local ENU.

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

    real_shape = lats.shape
    lats = lats.flatten()
    lons = lons.flatten()
    heights = heights.flatten()

    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)

    los_ecef = np.zeros((3, len(lats)))

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        # Geo2rdr is picky about how the heights look
        height_array = np.array(((height,),)).astype(np.double)

        # Set the target pixel location
        geo2rdr_obj.set_geo_coordinate(np.radians(lon), np.radians(lat), 1, 1, height_array)

        # compute the look vector and target-sensor range in ECEF
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        # LOS is defined as pointing from the ground pixel to the sensor
        los_ecef[:,i] = np.squeeze(geo2rdr_obj.get_los())

    # Sanity check for purpose of tracking problems
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

    return los_ecef.T


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
    '''
    TODO: docstring
    '''
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


