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

# def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
#    import Geo2rdr
#
#    real_shape = lats.shape
#    lats = lats.flatten()
#    lons = lons.flatten()
#    heights = heights.flatten()
#
#    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
#    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)
#
#    loss = np.zeros((3, len(lats)))
#    slant_ranges = np.zeros_like(lats)
#
#    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
#        height_array = np.array(((height,),))
#
#        # Geo2rdr is picky about the type of height
#        height_array = height_array.astype(np.double)
#
#        geo2rdr_obj.set_geo_coordinate(np.radians(lon),
#                                       np.radians(lat),
#                                       1, 1,
#                                       height_array)
#        # compute the radar coordinate for each geo coordinate
#        geo2rdr_obj.geo2rdr()
#
#        # get back the line of sight unit vector
#        los_x, los_y, los_z = geo2rdr_obj.get_los()
#        loss[:, i] = los_x, los_y, los_z
#
#        # get back the slant ranges
#        slant_range = geo2rdr_obj.get_slant_range()
#        slant_ranges[i] = slant_range
#
#    los = loss * slant_ranges
#
#    # Have to think about traversal order here. It's easy, though, since
#    # in both orders xs come first, followed by all ys, followed by all
#    # zs.
#    return los.reshape((3,) + real_shape)


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights, zref=_ZREF):
    '''
    Converts information from a state vector for a satellite orbit, given in terms of
    position and velocity, to line-of-sight information at each (lon,lat, height)
    coordinate requested by the user.

    *Note*:
    The LOS returned should be a vector pointing from the ground pixel to the sensor,
    truncating at the top of the troposphere, in an earth-centered, earth-fixed
    coordinate system.
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

    loss = np.zeros((3, len(lats)))
    slant_ranges = np.zeros_like(lats)

    # breakpoint()
    # lons = np.where(lons>=180, lons-360, lons) # 0:360 to -180:180
    pos  = lons>0                 # get bools of positive
    lons = lons % 360
    lons[lons == 0 & pos] == 360 # turn 0 back into 360

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        height_array = np.array(((height,),))

        # Geo2rdr is picky about the type of height
        height_array = height_array.astype(np.double)

        lon_start, lat_start = np.radians(lon), np.radians(lat)
        geo2rdr_obj.set_geo_coordinate(lon_start, lat_start, 1, 1, height_array)

        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los_x, los_y, los_z = geo2rdr_obj.get_los()
        loss[:, i] = ecef2enu(los_x, los_y, los_z, lat, lon, height)

        # get back the slant ranges
        slant_ranges[i] = geo2rdr_obj.get_slant_range()  #<- geo2rdr returns the slant range to sensor...not exactly what we want
        # print (slant_ranges[i])
        # //slant_range = sqrt(pow(xyz.x - st.position.x ,2) + pow(xyz.y - st.position.y, 2) + pow(xyz.z - st.position.z, 2));

        # x  -> sens
        # xe -> start

        # np.sqrt((x - xe)**2 + (y-ye)**2 + (z-ze)**2)

    # We need LOS defined as pointing from the ground pixel to the sensor in ECEF reference frame
    #sp = np.stack(utilFcns.lla2ecef(lats, lons, heights),axis = -1)
    #pt_rng = np.linalg.norm(sp,axis=-1)
    #slant_ranges = slant_ranges - pt_rng
    los = -loss  # * slant_ranges

    # Have to think about traversal order here. It's easy, though, since
    # in both orders xs come first, followed by all ys, followed by all
    # zs.
    return los.reshape(real_shape + (3,))


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
        time = (time - datetime.datetime(2018, 11, 11, 0, 0, 0)).total_seconds()

        time = time - t[0]

    t  = t - t[0]

    idx = cut_orbit_file(times)
    t, x, y, z, vx, vy, vz = t[idx], x[idx], y[idx], z[idx], vx[idx], vy[idx], vz[idx]

    # if time is not None:
    #    mask = np.abs(t - time) < 3600
    #    t, x, y, z, vx, vy, vz = t[mask], x[mask], y[mask], z[mask], vx[mask], vy[mask], vz[mask]

    return [t, x, y, z, vx, vy, vz]


def infer_sv(los_file, lats, lons, heights, time=None):
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
    LOSs = state_to_los(*svs, lats=lats, lons=lons, heights=heights)
    return LOSs


def los_to_lv(incidence, heading, lats, lons, heights, zref, ranges=None):
    '''
    Convert incidence and heading to line-of-sight vectors from the ground to the top of
    the troposphere.

    *NOTE*:
    LOS here is defined in an Earth-centered, earth-referenced
    coordinate system as pointing from the ground pixel to the sensor, truncating at the top of
    the troposphere.

    Algorithm referenced from http://earthdef.caltech.edu/boards/4/topics/327
    '''
    a_0 = incidence
    a_1 = heading

    east = utilFcns.sind(a_0) * utilFcns.cosd(a_1 + 90)
    north = utilFcns.sind(a_0) * utilFcns.sind(a_1 + 90)
    up = utilFcns.cosd(a_0)
    los = np.stack((east, north, up), axis=-1)
    return los

    # # Pick reasonable range to top of troposphere if not provided
    # if ranges is None:
    #     ranges = (zref - heights) / up
    # #slant_range = ranges = (zref - heights) / utilFcns.cosd(inc)
    #
    # # Scale look vectors by range
    # east, north, up = np.stack((east, north, up)) #* ranges
    #
    # xyz = utilFcns.enu2ecef(
    #     east.flatten(), north.flatten(), up.flatten(), lats.flatten(),
    #     lons.flatten(), heights.flatten())
    #
    # sp_xyz = utilFcns.lla2ecef(lats.flatten(), lons.flatten(), heights.flatten())
    # los = np.stack(xyz, axis=-1) - np.stack(sp_xyz, axis=-1)
    # los = los.reshape(east.shape + (3,))

    return los


def infer_los(los, lats, lons, heights, zref, time=None):
    '''
    Helper function to deal with various LOS files supplied
    '''

    los_type, los_file = los

    if los_type == 'sv':
        LOS = infer_sv(los_file, lats, lons, heights, time)
        np.save('./HR_SV/LOS.npy', LOS)
        print ('Saved los SV file')
    elif los_type == 'los':
        incidence, heading = [f.flatten() for f in utilFcns.gdal_open(los_file)]
        utilFcns.checkShapes(np.stack((incidence, heading), axis=-1), lats, lons, heights)
        LOS = los_to_lv(incidence, heading, lats, lons, heights, zref)
        np.save('./HR_LOS/LOS.npy', LOS)
        print ('Saved los LOS file')
    else:
        raise ValueError("Unsupported los type '{}'".format(los_type))
    return LOS


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


def getLookVectors(look_vecs, lats, lons, heights, zref=_ZREF, time=None):
    '''
    If the input look vectors are specified as Zenith, compute and return the
    look vectors. Otherwise, check that the look_vecs shape makes sense.
    '''
    if look_vecs is None:
        look_vecs = Zenith

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
