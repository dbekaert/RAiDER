#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer
#  Copyright 2020. ALL RIGHTS RESERVED.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import pathlib
import numpy as np

from abc import ABC, abstractmethod
from typing import NamedTuple

from RAiDER import Geo2rdr
from RAiDER.constants import Zenith
from RAiDER.losreader import (
    read_ESA_Orbit_file, read_los_file, read_shelve, read_txt_file
)
from RAiDER.utilFcns import cosd, enu2ecef, gdal_open, lla2ecef, sind

'''
*Note*:
The line-of-sight look vector should be a unit-length 3-component vector 
pointing from the ground pixel to the sensor. The associated projection 
is earth-centered, earth-fixed.
'''

class Points(NamedTuple('Points', ['lats', 'lons', 'hgts'])):
    __slots__ = ()


class LVGenerator(ABC):
    """Look vector generator"""

    @abstractmethod
    def generate(self, llh):
        """
        Generate look vectors for a set of locations

        :param llh: 3d numpy array of pixel locations in LLA coordinates.
        :return: 3d numpy array of unit look vectors at each location in ECEF
            coordinates.
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
        states  - Orbital state vectors
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

    # TODO: This method of passing in the los mode is really ugly
    if los_mode is not Zenith and los_mode[0] == "los":
        # We already have state vectors so we just need to transform them into
        # the right coordinate system
        a_0, a_1 = read_los_file(los_mode[1])
        los = incidence_heading_to_los(a_0, a_1)
        if los.shape != llh.shape[:-1]:
            raise ValueError(
                "Incidence/heading values had wrong shape! Incidence/heading "
                "shape {}, coordinate shape {}".format(
                    los.shape, llh.shape[:-1]
                )
            )
        return los

    gen = get_lv_generator(los_mode)
    look_vectors = gen.generate(llh)

    lats, lons, heights = llh[..., 0], llh[..., 1], llh[..., 2]

    mask = np.isnan(heights) | np.isnan(lats) | np.isnan(lons)
    look_vectors[mask, :] = np.nan

    return look_vectors


def incidence_heading_to_los(a_0, a_1):
    """
    Convert incidence-heading information into unit vectors in ECEF coordinates.
    """
    assert a_0.shape == a_1.shape, "Incompatible dimensions!"

    east = sind(a_0) * cosd(a_1 + 90)
    north = sind(a_0) * sind(a_1 + 90)
    up = cosd(a_0)

    return np.stack((east, north, up), axis=-1)


def get_lv_generator(los_mode):
    """
    Convert los_mode to a look vector generator.

    TODO: Really this should be done much higher up in the call chain.
    """
    if los_mode is Zenith:
        return ZenithLVGenerator()

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

    raise ValueError("los_type '{}' is not supported!".format(los_type))
