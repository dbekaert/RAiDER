#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
# 
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

import numpy as np
import os.path
import shelve
import RAiDER.utilFcns as utilFcns
from RAiDER.constants import _ZREF, Zenith


#TODO: It looks like Configurable and ProductManager are not functioning
class Configurable():
    '''
    Is this needed? 
    '''
    def __init__(self):
        pass

class ProductManager(Configurable):
    '''
    TODO: docstring
    '''
    family = 'productmanager'
    def __init__(self,family='', name=''):
        super(ProductManager, self).__init__(family if family else  self.__class__.family, name=name)
        
    def dumpProduct(self,obj,filename):
        self._instance = obj
        self.dump(filename)
        
    def loadProduct(self,filename):
        self.load(filename)
        return self._instance


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights, zref = _ZREF):
    '''
    Converts information from a state vector for a satellite orbit, given in terms of 
    position and velocity, to line-of-sight information at each (lon,lat, height) 
    coordinate requested by the user.

    *Note*: 
    The LOS returned should be a vector pointing from the ground pixel to the sensor, 
    truncating at the top of the troposphere, in an earth-centered, earth-fixed 
    coordinate system. 
    '''
    from RAiDER import Geo2rdr

    # check the inputs
    if t.size < 4:
        raise RuntimeError('state_to_los: At least 4 state vectors are required for orbit interpolation')
    if t.shape!=x.shape:
        raise RuntimeError('state_to_los: t and x must be the same size')
    if lats.shape!=lons.shape:
        raise RuntimeError('state_to_los: lats and lons must be the same size')

    real_shape = lats.shape
    lats = lats.flatten()
    lons = lons.flatten()
    heights = heights.flatten()

    geo2rdr_obj = Geo2rdr.PyGeo2rdr()
    geo2rdr_obj.set_orbit(t, x, y, z, vx, vy, vz)

    loss = np.zeros((3, len(lats)))
    slant_ranges = np.zeros_like(lats)
 

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        height_array = np.array(((height,),))

        # Geo2rdr is picky about the type of height
        height_array = height_array.astype(np.double)

        lon_start, lat_start = np.radians(360-lon),np.radians(lat)
        geo2rdr_obj.set_geo_coordinate(lon_start, lat_start,1, 1,height_array)

        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los_x, los_y, los_z = geo2rdr_obj.get_los()
        loss[:, i] = los_x, los_y, los_z

        # get back the slant ranges
        slant_range = geo2rdr_obj.get_slant_range()  #<- geo2rdr returns the slant range to sensor...not exactly what we want
        slant_ranges[i] = slant_range

    los = loss * slant_ranges
    #los = loss #<- just keep in unit vector form?

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
        t[i] = st.time.second + st.time.minute*60.0
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
    return [np.array(a) for a in [t,x,y,z, vx,vy, vz]]


def read_ESA_Orbit_file(filename):
    '''
    Read orbit data from an orbit file supplied by ESA
    '''
    import datetime
    import xml.etree.ElementTree as ET

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

    for i, st in enumerate(data_block[0]):
        t[i] = (datetime.datetime.strptime(st[1].text, 'UTC=%Y-%m-%dT%H:%M:%S.%f') - datetime.datetime(1970,1,1)).total_seconds()
        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i]= float(st[7].text)
        vy[i]= float(st[8].text)
        vz[i]= float(st[9].text)

    t = t - t[0]

    return [t, x, y, z, vx, vy, vz]


def read_xml_file(filename):

    pm = ProductManager()
    pm.configure()

    obj = pm.loadProduct(filename)

    numSV = len(obj.orbit.stateVectors)

    t = np.ones(numSV)
    x = np.ones(numSV)
    y = np.ones(numSV)
    z = np.ones(numSV)
    vx = np.ones(numSV)
    vy = np.ones(numSV)
    vz = np.ones(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        t[i] = st.time.second + st.time.minute*60.0
        x[i] = st.position[0]
        y[i] = st.position[1]
        z[i] = st.position[2]
        vx[i] = st.velocity[0]
        vy[i] = st.velocity[1]
        vz[i] = st.velocity[2]

    return t, x, y, z, vx, vy, vz


def infer_sv(los_file, lats, lons, heights):
    """Read an LOS file."""
    # TODO: Change this to a try/except structure
    _, ext = os.path.splitext(los_file)
    if ext == '.txt':
        svs = read_txt_file(los_file)
    elif ext == '.xml':
        svs = read_xml_file(los_file)
    elif ext == '.EOF':
        svs = read_ESA_Orbit_file(los_file)
    else:
        # Here's where things get complicated... Either it's a shelve
        # file or the user messed up. For now we'll just try to read it
        # as a shelve file, and throw whatever error that does, although
        # the message might be sometimes misleading.
        svs = read_shelve(los_file)
    LOSs = state_to_los(*svs, lats = lats, lons = lons, heights = heights)
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

    east = utilFcns.sind(a_0)*utilFcns.cosd(a_1 + 90)
    north = utilFcns.sind(a_0)*utilFcns.sind(a_1 + 90)
    up = utilFcns.cosd(a_0)
    east, north, up = np.stack((east, north, up))

    # Pick reasonable range to top of troposphere if not provided
    if ranges is None:
        ranges = (zref - heights) / up
    #slant_range = ranges = (zref - heights) / utilFcns.cosd(inc)

    # Scale look vectors by range
    east, north, up = np.stack((east, north, up)) * ranges

    xyz = utilFcns.enu2ecef(
            east.flatten(), north.flatten(), up.flatten(), lats.flatten(),
            lons.flatten(), heights.flatten())

    sp_xyz = utilFcns.lla2ecef(lats.flatten(), lons.flatten(), heights.flatten())
    los = np.stack(xyz, axis=-1) - np.stack(sp_xyz, axis=-1)
    los = los.reshape(east.shape + (3,))

    return los


def infer_los(los, lats, lons, heights, zref):
    '''
    Helper function to deal with various LOS files supplied
    '''

    los_type, los_file = los

    if los_type == 'sv':
        LOS = infer_sv(los_file, lats, lons, heights)

    if los_type == 'los':
        from RAiDER.utilFcns import checkShapes
        incidence, heading = [f.flatten() for f in utilFcns.gdal_open(los_file)]
        checkShapes(np.stack((incidence, heading), axis = -1), lats, lons, heights)
        LOS = los_to_lv(incidence, heading, lats, lons, heights, zref)

    return LOS

    raise ValueError("Unsupported los type '{}'".format(los_type))


def _getZenithLookVecs(lats, lons, heights, zref = _ZREF):

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
        if (lats.ndim!=1) | (heights.ndim!=1) | (lons.ndim!=1):
            raise RuntimeError('_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays')
    except AttributeError:
        raise RuntimeError('_getZenithLookVecs: lats/lons/heights must be 1-D numpy arrays')
    if hasattr(zref, "__len__") | isinstance(zref, str):
        raise RuntimeError('_getZenithLookVecs: zref must be a scalar')

    e = np.cos(np.radians(lats))*np.cos(np.radians(lons))
    n = np.cos(np.radians(lats))*np.sin(np.radians(lons))
    u = np.sin(np.radians(lats))
    zenLookVecs = (np.array((e,n,u)).T*(zref - heights)[..., np.newaxis])
    return zenLookVecs


def getLookVectors(look_vecs, lats, lons, heights, zref = _ZREF):
    '''
    If the input look vectors are specified as Zenith, compute and return the
    look vectors. Otherwise, check that the look_vecs shape makes sense. 
    '''
    if look_vecs is None:
        look_vecs= Zenith

    in_shape = lats.shape
    lat = lats.flatten()
    lon = lons.flatten()
    hgt = heights.flatten()

    if look_vecs is Zenith:
        look_vecs = _getZenithLookVecs(lat, lon, hgt, zref = zref)
    else:
        look_vecs = infer_los(look_vecs, lat, lon, hgt, zref)

    mask = np.isnan(hgt) | np.isnan(lat) | np.isnan(lon)
    look_vecs[mask,:] = np.nan

    return look_vecs.reshape(in_shape + (3,))



def dt2float(date):
    import datetime as dt
    import time

    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    # check that the object is a datetime
    try:
        year = date.year
    except AttributeError:
        date = numpyDT64ToDatetime(date)
        year = date.year

    startOfThisYear = dt.datetime(year=year, month=1, day=1)
    startOfNextYear = dt.datetime(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration
    date_frac = date.year + fraction

    return date_frac
