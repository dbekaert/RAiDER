#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer
#  Copyright 2020. ALL RIGHTS RESERVED.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from abc import ABC, abstractmethod
from RAiDER.constants import Zenith, Slant
from RAiDER.utilFcns import gdal_open, sind, cosd


class Points():
    ''' A class object to store point locations '''
    def __init__(self, llh):
        self.lats = None 
        self.lons = None 
        self.hgts = None 

    def setLLH(self, llh):
        self.lats = llh[...,0]
        self.lons = llh[...,1]
        self.hgts = llh[...,2]

    def getLLH(self, llh):
        return self.lats, self.lons, self.hgts

    def _checkLLH(self):
        if self.lats is None:
            raise ValueError(
                'Please set the point locations using "<self>.setLLH(llh)"'
            )


class LookVector(ABC):
    ''' A base class for RADAR look vectors '''
    def __init__(self):
        self.points = None  #TODO: should this be "points" or lat/lon/hgt separately? 
        self.vectors = None #TODO: vectors should be in earth-centered, earth-fixed reference frame, but "points" are in LLH
        self._proj = None

    @abstractmethod
    def setVectors(self):
        pass

    def setPoints(self, lats, lons, hgts):
        self.lats = lats 
        self.lons = lons 
        self.hgts = hgts 

    def transform(self, new_proj):
        pass # TODO: implement this base class method


class Zenith(LookVector):
    """A Zenith look vector."""
    def __init__(self):
        LookVector.__init__(self)

    def setVectors(self):
        self._checkLLH()
        self.los = zenithLookVectors(llh)


class Slant(LookVector):
    """A slant (i.e., true line-of-sight) look vector."""
    def __init__(self):
        LookVector.__init__(self)
        self.reader = None



#TODO: figure out how to best call the readers from the LookVector Object
class Reader(ABC):
    ''' Generic object for reading look vectors from files '''
    def __init__(self):
        self.lv = None
        self.points = None

    #TODO: Need to set the points using the Points object

    def ReadVectors(self, filename):
        self._checkLLH()
        self._read(filename)
        return self.lv

    @abstractmethod
    def _read(self, filename):
        pass


class RasterReader(Reader):
    '''
    Get line-of-sight vectors from an ISCE dual-band raster file
    containing inclination (band 1) and heading (band 2) information
    '''
    def __init__(self):
        Reader.__init__(self)

    def _read(self, filename)
        inc, hd = [f for f in gdal_open(filename)]
        east = sind(inc) * cosd(hd + 90)
        north = sind(inc) * sind(hd + 90)
        up = cosd(inc)

        # ensure shape compatibility
        if up.shape != self.lats.shape:
            east = east.flatten()
            north = north.flatten()
            up = up.flatten()
        if east.shape != self.lats.flatten().shape:
            raise ValueError(
                'The number or shape of the input LOS vectors is different ' 
                'from the ground pixels, please check  your inputs'
            )

        # Convert unit vectors to Earth-centered, earth-fixed
        if self.lats is not None:
            self.lv = enu2ecef(
                east, 
                north, 
                up, 
                self.lats, 
                self.lons, 
                self.hgts
            )
        else: 
            raise ValueError(
                'You need to assign pixel locations. Use <self>.setLLH(llh)'
            )


class OrbitFileReader(Reader):
    '''
    Get line-of-sight vectors from an ESA orbit file
    '''
    def __init__(self, llh, filename):
        Reader.__init__(self, llh)

    def _read(self, filename)
        svs = read_ESA_Orbit_file(filename)
        self.lv = state_to_los(svs, self.lats, self.lons, self.hgts)
 

def zenithLookVectors(llh):
    '''
    Returns look vectors when Zenith is used.
    Inputs:
       llh   - ... x 3 numpy array numpy of pixel locations
       zref  - float, integration height in meters
    Outputs:
       los   - an Nx3 numpy array with the look vectors.
    '''
    lats = llh[...,0]
    lons = llh[...,1]
    hgts = llh[...,2]

    e = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    n = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    u = np.sin(np.radians(lats))

    return np.array((e, n, u)).T


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

    for i, st in enumerate(data_block[0]):
        t[i] = (datetime.datetime.strptime(st[1].text, 'UTC=%Y-%m-%dT%H:%M:%S.%f') - datetime.datetime(1970, 1, 1)).total_seconds()
        x[i] = float(st[4].text)
        y[i] = float(st[5].text)
        z[i] = float(st[6].text)
        vx[i] = float(st[7].text)
        vy[i] = float(st[8].text)
        vz[i] = float(st[9].text)

    # Get the reference time
    if time is not None:
        time = (time - datetime.datetime(1970, 1, 1)).total_seconds()
        time = time - t[0]

    t = t - t[0]

    return [t, x, y, z, vx, vy, vz]


def state_to_los(svs, lats, lons, heights):
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
    geo2rdr_obj.set_orbit(*svs)

    los = np.zeros((3, len(lats)))
    slant_ranges = np.zeros_like(lats)

    for i, (lat, lon, height) in enumerate(zip(lats, lons, heights)):
        height_array = np.array(((height,),))

        # Geo2rdr is picky about the type of height
        height_array = height_array.astype(np.double)

        lon_start, lat_start = np.radians(360 - lon), np.radians(lat)
        geo2rdr_obj.set_geo_coordinate(
            np.radians(lon),
           np.radians(lon),
            np.radians(lat),
            1, 1,
            height_array
        )

        # compute the radar coordinate for each geo coordinate
        geo2rdr_obj.geo2rdr()

        # get back the line of sight unit vector
        los[:, i] = geo2rdr_obj.get_los()

    los = los.T.reshape(real_shape + (3,))

    return los


def getLookVectors(look_vecs, llh):
    '''
    Returns unit look vectors for each query point specified as a lat/lon/height.
    Inputs:
        look_vecs    - Can be a Zenith object, a two-band file containing line-of-
                       sight vectors (inclination, heading), or an ESA orbit file 
                       for the time period of interest. 
        llh          - latitude, longitude, heights for the query points

    Returns: 
        Unit look vectors pointing from each ground point towards the sensor or 
        Zenith
    '''
    pass
    

