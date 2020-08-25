#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer
#  Copyright 2020. ALL RIGHTS RESERVED.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pathlib
from abc import ABC, abstractmethod

import numpy as np

from RAiDER import Geo2rdr
from RAiDER.constants import Zenith, _ZREF
from RAiDER.losreader import read_ESA_Orbit_file, read_shelve, read_txt_file
from RAiDER.utilFcns import cosd, enu2ecef, gdal_open, lla2ecef, sind

'''
*Note*:
The line-of-sight look vector should be a unit-length 3-component vector 
pointing from the ground pixel to the sensor. The associated projection 
is earth-centered, earth-fixed.
'''

class Points(NamedTuple):
    ''' A class object to store point locations '''
    def __init__(self, llh):
        self.lats = llh[..., 0]
        self.lons = llh[..., 1]
        self.hgts = llh[..., 2]


class LVGenerator(ABC):
    """Look vector generator"""

    @abstractmethod
    def generate(self, llh):
        """
        Generate look vectors for a set of locations

        :param llh: 3d numpy array of pixel locations in llh coordinates.
        :return: 3d numpy array of look vectors at teach location.
        """
        ...

    def _calculate_lengths(self, enu):
        ''' 
        Calculate the length from the ground pixel to the reference 
        height along the look direction specified by the unit look
        vector los.
        '''
        return self.zref/(np.sqrt(1 - 1/(np.square(enu[..., 0]) + np.square(enu[...,1]))))


class ZenithLVGenerator(LVGenerator):
    """Generate look vectors pointing towards the zenith"""

    def __init__(self, zref=_ZREF):
        """
        zref  - float, integration height in meters
        """
        self.zref = zref
        self.los = None
        self.length = None

    def generate(self, llh):
        '''
        Returns look vectors when Zenith is used.
        Inputs:
           llh   - ... x 3 numpy array numpy of pixel locations
        Outputs:
           los   - an Nx3 numpy array with the look vectors.
        '''
        lats = llh[..., 0]
        lons = llh[..., 1]
        hgts = llh[..., 2]

        e = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
        n = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
        u = np.sin(np.radians(lats))

        lengths = self._calculate_length(enu)

        los = enu2ecef(
            e.ravel(), 
            n.ravel(), 
            u.ravel(), 
            lats.ravel(),
            lons.ravel(), 
            hgts.ravel()
        )

        return los.reshape(e.shape + (3,)), lengths.reshape(e.shape)


class OrbitLVGenerator(LVGenerator):
    """Generate look vectors from orbital state information"""

    def __init__(self, states):
        """
        states  -
        """
        self.states = states

    def generate(self, llh):
        '''
        Converts information from a state vector for a satellite orbit, given
        in terms of position and velocity, to line-of-sight information at each
        (lon,lat, height) coordinate requested by the user.
        '''
        assert self.states.t.size >= 4, \
            "At least 4 state vectors are required for orbit interpolation"
        assert self.states.t.shape == self.states. x.shape, \
            "t and x must be the same size"

        real_shape = llh.shape[:-1]
        lats = llh[..., 0].flatten()
        lons = llh[..., 1].flatten()
        heights = llh[..., 2].flatten()

        geo2rdr_obj = Geo2rdr.PyGeo2rdr()
        geo2rdr_obj.set_orbit(*self.states)

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

        

        return los, lengths


class IHLVGenerator(LVGenerator):
    """Generate look vectors from incidence and heading information"""

    def __init__(self, incidence, heading, zref, ranges=None):
        """
        incidence  -
        heading    -
        zref       - float, integration height in meters
        ranges     -
        """
        assert incidence.shape == heading.shape, \
            "Incidence and heading must have the same shape!"
        self.incidence = incidence
        self.heading = heading
        self.zref = zref
        self.ranges = ranges

    def generate(self, llh):
        """
        Convert incidence and heading to line-of-sight vectors from the ground
        to the top of the troposphere.

        *NOTE*:
        LOS here is defined in an Earth-centered, earth-referenced
        coordinate system as pointing from the ground pixel to the sensor,
        truncating at the top of the troposphere.

        Algorithm referenced from http://earthdef.caltech.edu/boards/4/topics/327
        """
        lats, lons, heights = llh[..., 0], llh[..., 1], llh[..., 2]
        a_0 = self.incidence
        a_1 = self.heading
        ranges = self.ranges
        zref = self.zref

        if self.incidence.shape != heights.shape:
            raise ValueError(
                "Incidence/heading values had wrong shape! Incidence shape "
                "{} Heading shape {} coordinate shape {}".format(
                    self.incidence.shape,
                    self.heading.shape,
                    heights.shape
                )
            )

        east = sind(a_0) * cosd(a_1 + 90)
        north = sind(a_0) * sind(a_1 + 90)
        up = cosd(a_0)
        east, north, up = np.stack((east, north, up))

        # Pick reasonable range to top of troposphere if not provided
        if ranges is None:
            ranges = (zref - heights) / up
        # slant_range = ranges = (zref - heights) / utilFcns.cosd(inc)

        # Scale look vectors by range
        east, north, up = np.stack((east, north, up)) * ranges

        xyz = enu2ecef(
            east.ravel(), north.ravel(), up.ravel(), lats.ravel(),
            lons.ravel(), heights.ravel()
        )

        sp_xyz = lla2ecef(lats.ravel(), lons.ravel(), heights.ravel())
        los = np.stack(xyz, axis=-1) - np.stack(sp_xyz, axis=-1)
        los = los.reshape(east.shape + (3,))

        return los


def getLookVectors(los_mode, llh):
    '''
    Returns unit look vectors for each query point specified as a lat/lon/height.
    Inputs:
        los_mode     - Can be a Zenith object, a two-band file containing line-of-
                       sight vectors (inclination, heading), or an ESA orbit file
                       for the time period of interest.
        llh          - latitude, longitude, heights for the query points

    Returns:
        Unit look vectors pointing from each ground point towards the sensor or
        Zenith
    '''
    if los_mode is None:
        los_mode = Zenith

    gen = get_lv_generator(los_mode)
    look_vectors = gen.generate(llh)

    lats, lons, heights = llh[..., 0], llh[..., 1], llh[..., 2]

    mask = np.isnan(heights) | np.isnan(lats) | np.isnan(lons)
    look_vectors[mask, :] = np.nan

    return look_vectors


def get_lv_generator(los_mode):
    if los_mode is Zenith:
        return ZenithLVGenerator()

    # TODO: Do we actually need this type flag here or is it always
    # unambiguous from the file extension?
    los_type, filepath = los_mode

    if los_type == "sv":
        # Using orbital state information
        ext = pathlib.Path(filepath).suffix

        reader_func = {
            ".txt": read_txt_file,
            ".eof": read_ESA_Orbit_file
        }.get(ext.lower()) or read_shelve

        states = reader_func(filepath)
        return OrbitLVGenerator(states)

    if los_type == "los":
        # Using incidence and heading information
        incidence, heading = [f.flatten() for f in gdal_open(filepath)]
        if incidence.shape != heading.shape:
            raise ValueError(
                "Malformed los file. Incidence shape {} and heading shape {} "
                "do not match!".format(incidence.shape, heading.shape)
            )

        return IHLVGenerator(incidence, heading)
