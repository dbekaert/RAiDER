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
from RAiDER.constants import Zenith, _ZREF
from RAiDER.losreader import (
    read_ESA_Orbit_file, read_los_file, read_shelve, read_txt_file
)
from RAiDER.utilFcns import cosd, sind, enu2ecef, gdal_open, lla2ecef

'''
*Note*:
The line-of-sight look vector should be a unit-length 3-component vector 
pointing from the ground pixel to the sensor. The associated projection 
is earth-centered, earth-fixed.
'''


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

    def getLOSType(self):
        return self._los_type


class LOSGenerator(LVGenerator):
    """ A dummy class for holding line-of-sight vectors from rasters """

    def __init__(self, los):
        self._los_type = 'STD'
        self._los = los

    def generate(self, llh):
        if self._los.shape != llh.shape:
            raise ValueError(
                "The shape of the input lat/lon points didn't match "
                "the line-of-sight vector. llh shape: {}, los shape: {}"
                .format(los.shape, llh.shape)
            )
        return self._los


class ZenithLVGenerator(LVGenerator):
    """Generate look vectors pointing towards the zenith"""

    def __init__(self):
        """
        zref  - float, integration height in meters
        """
        self._los_type = 'ZTD'

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

        e = cosd(lats) * cosd(lons)
        n = cosd(lats) * sind(lons)
        u = sind(lats)

        return np.stack([e, n, u], axis=-1)


class OrbitLVGenerator(LVGenerator):
    """Generate look vectors from orbital state information"""

    def __init__(self, states):
        """
        states  - Orbital state vectors
        """
        self._los_type = 'STD'
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

        return los


def getLookVectors(generator, llh):
    '''
    Returns unit look vectors for each query point specified as a lat/lon/height.
    Inputs:
        generator    - Can be a Zenith object, a two-band file containing line-of-
                       sight vectors (inclination, heading), or an ESA orbit file
                       for the time period of interest.
        llh          - latitude, longitude, heights for the query points

    Returns:
        Unit look vectors pointing from each ground point towards the sensor or
        Zenith
        The length of rays in the vector directions
    '''
    look_vectors = generator.generate(llh)
    mask = np.isnan(np.mean(llh, axis=-1))
    look_vectors[mask, :] = np.nan

    return look_vectors


def calculate_ray_lengths(los, zref):
    ''' 
    Calculate the length of the ray paths (for non-Zenith only)
    '''
    return zref / (np.sqrt(1 - 1 / (np.square(los[..., 0]) + np.square(los[..., 1]))))
