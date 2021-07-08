#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
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


def getLookVectors(look_vecs, lats, lons, heights, zref=_ZREF, time=None):
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
        look_vecs = _getZenithLookVecs(lat, lon, hgt, zref=zref)
    else:
        look_vecs = infer_los(look_vecs, lat, lon, hgt, zref, time)

    mask = np.isnan(hgt) | np.isnan(lat) | np.isnan(lon)
    look_vecs[mask, :] = np.nan

    return look_vecs.reshape(in_shape + (3,)).astype(np.float64)


def _getZenithLookVecs(lats, lons, heights, zref=_ZREF):
    '''
    Returns look vectors when Zenith is used.
    Inputs:
       lats/lons/heights - Nx1 numpy arrays of points.
       zref              - float, integration height in meters
    Outputs:
       zenLookVecs       - an Nx3 numpy array with the look vectors.
                           The vectors give the zenith ray paths for
                           each of the points to the top of the atmosphere.
    '''
    try:
        if (lats.ndim != 1) | (heights.ndim != 1) | (lons.ndim != 1):
            raise RuntimeError('_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays')
    except AttributeError:
        raise RuntimeError('_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays')
    if hasattr(zref, "__len__") | isinstance(zref, str):
        raise RuntimeError('_getZenithLookVecs: zref must be a scalar')

    e = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    n = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    u = np.sin(np.radians(lats))
    zenLookVecs = (np.array((e, n, u)).T * (zref - heights)[..., np.newaxis])
    return zenLookVecs.astype(np.float64)


def infer_los(los_file, lats, lons, heights, zref, time=None):
    '''
    Helper function to deal with various LOS files supplied
    '''
    breakpoint()
    try:
        incidence, heading = [f.flatten() for f in utilFcns.gdal_open(los_file)]
        utilFcns.checkShapes(np.stack((incidence, heading), axis=-1), lats, lons, heights)
        LOS_enu = los_to_lv(incidence, heading, lats, lons, heights, zref)
        LOS = utilFcns.enu2ecef(LOS_enu[...,0], LOS_enu[...,1], LOS_enu[...,2], lats, lons, heights)

    except OSError:
        svs = get_sv(los_file, lats, lons, heights, time)
        LOS = state_to_los(*svs, lats=lats, lons=lons, heights=heights)

    return LOS


def get_sv(los_file, lats, lons, heights, time=None):
    """Read an LOS file."""
    # TODO: Change this to a try/except structure
    _, ext = os.path.splitext(los_file)
    if ext == '.txt':
        svs = read_txt_file(los_file)
    elif ext == '.EOF':
        svs = read_ESA_Orbit_file(los_file, time)
    else:
        # Here's where things get complicated... Either it's a shelve
        # file or the user messed up. For now we'll just try to read it
        # as a shelve file, and throw whatever error that does, although
        # the message might be sometimes misleading.
        svs = read_shelve(los_file)
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
        raise RuntimeError('state_to_los: At least 4 state vectors are required for orbit interpolation')
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

    return los_ecef.T


def cut_orbit_file(times, ref_time_st=None, ref_time_en=None):
    """ Slice the orbit file around the reference aquisition time """
    # eventually refs will have to be gotten from SLCs which we don't require
    pad = 2 # minutes
    # round to nearest minute and pad
    ref_time_st = utilFcns.round_time(datetime.datetime(2018, 11, 13, 23, 6, 17), 60) \
                                     - datetime.timedelta(minutes=pad)
    ref_time_en = utilFcns.round_time(datetime.datetime(2018, 11, 13, 23, 6, 44), 60) \
                                     + datetime.timedelta(minutes=pad)

    idx, tim_close = [], []
    # iterate through times in orbit file, finding those within padded span
    for i, time in enumerate(times):
        if ref_time_st <= time <= ref_time_en:
            idx.append(i)
            tim_close.append(time)
    return idx


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


def read_ESA_Orbit_file(filename, time=None):
    '''
    Read orbit data from an orbit file supplied by ESA
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
        t[i] = (datetime.datetime.strptime(st[1].text, 'UTC=%Y-%m-%dT%H:%M:%S.%f')
                    # - datetime.datetime(1970, 1, 1)).total_seconds()
                    - datetime.datetime(2018, 11, 11, 0, 0, 0)).total_seconds()

        times.append(datetime.datetime.strptime(st[1].text, 'UTC=%Y-%m-%dT%H:%M:%S.%f'))
        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i] = float(st[7].text)
        vy[i] = float(st[8].text)
        vz[i] = float(st[9].text)

    # Get the reference time
    if time is not None:
        raise RuntimeError('Need to finish this')
        mask = np.abs(t - time) < 100 # Need syntax to compare seconds
        t, x, y, z, vx, vy, vz = t[mask], x[mask], y[mask], z[mask], vx[mask], vy[mask], vz[mask]
    else:
        idx = cut_orbit_file(times)
        t, x, y, z, vx, vy, vz = t[idx], x[idx], y[idx], z[idx], vx[idx], vy[idx], vz[idx]

    return [t, x, y, z, vx, vy, vz]


