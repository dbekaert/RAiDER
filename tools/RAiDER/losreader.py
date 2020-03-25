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


def state_to_los(t, x, y, z, vx, vy, vz, lats, lons, heights):
    from RAiDER import Geo2rdr

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

        geo2rdr_obj.set_geo_coordinate(np.radians(lon),
                                       np.radians(lat),
                                       1, 1,
                                       height_array)
        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los_x, los_y, los_z = geo2rdr_obj.get_los()
        loss[:, i] = los_x, los_y, los_z

        # get back the slant ranges
        slant_range = geo2rdr_obj.get_slant_range()
        slant_ranges[i] = slant_range

    los = loss * slant_ranges

    # Have to think about traversal order here. It's easy, though, since
    # in both orders xs come first, followed by all ys, followed by all
    # zs.
    return los.reshape((3,) + real_shape)


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
                t_, x_, y_, z_, vx_, vy_, vz_ = line.split()
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
    return (np.array(t), np.array(x), np.array(y), np.array(z), np.array(vx),
            np.array(vy), np.array(vz))


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
    """Infer the type of file to read, then read an LOS file."""
    _, ext = os.path.splitext(los_file)
    if ext == '.txt':
        svs = read_txt_file(los_file)
    elif ext == '.xml':
        svs = read_xml_file(los_file)
    else:
        # Here's where things get complicated... Either it's a shelve
        # file or the user messed up. For now we'll just try to read it
        # as a shelve file, and throw whatever error that does, although
        # the message might be sometimes misleading.
        svs = read_shelve(los_file)
    LOSs = state_to_los(*svs, lats = lats, lons = lons, heights = heights)
    return LOSs


def los_to_lv(incidence, heading, lats, lons, heights, zref, ranges=None):
    # I'm looking at http://earthdef.caltech.edu/boards/4/topics/327
    a_0 = incidence
    a_1 = heading

    east = utilFcns.sind(a_0)*utilFcns.cosd(a_1 + 90)
    north = utilFcns.sind(a_0)*utilFcns.sind(a_1 + 90)
    up = utilFcns.cosd(a_0)

    east, north, up = np.stack((east, north, up))

    # Pick reasonable range to top of troposphere if not provided
    if ranges is None:
        ranges = (zref - heights) / up

    # Scale look vectors by range
    east, north, up = np.stack((east, north, up)) * ranges

    x, y, z = utilFcns.enu2ecef(
            east.flatten(), north.flatten(), up.flatten(), lats.flatten(),
            lons.flatten(), heights.flatten())

    los = (np.stack((x, y, z), axis=-1) - np.stack(utilFcns.lla2ecef(
               lats.flatten(), lons.flatten(), heights.flatten()), axis=-1))
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
    import pdb; pdb.set_trace()

    mask = np.isnan(hgt) | np.isnan(lat) | np.isnan(lon)
    look_vecs[mask,:] = np.nan

    # check size
    if look_vecs.ndim==1:
       if len(look_vecs)!=3:
          raise RuntimeError('look_vecs must be Nx3') 
    if look_vecs.shape[-1]!=3:
       raise RuntimeError('look_vecs must be Nx3')

    return look_vecs.reshape(in_shape + (3,))



