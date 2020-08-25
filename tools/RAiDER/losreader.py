#!/usr/bin/env python3
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#  Author: Jeremy Maurer, Raymond Hogenson & David Bekaert
#  Copyright 2019, by the California Institute of Technology. ALL RIGHTS
#  RESERVED. United States Government Sponsorship acknowledged.
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime
import shelve
from typing import NamedTuple
from xml.etree import ElementTree as ET

import numpy as np

from RAiDER.utilFcns import gdal_open


class OrbitStates(NamedTuple):
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    vz: np.ndarray

    @classmethod
    def empty(cls, num=0):
        return cls(*(np.ones(num) for _ in range(len(cls._fields))))


def read_shelve(filename):
    '''
    TODO: docstring
    '''
    with shelve.open(filename, 'r') as db:
        obj = db['frame']

    numSV = len(obj.orbit.stateVectors)

    states = OrbitStates.empty(numSV)

    for i, st in enumerate(obj.orbit.stateVectors):
        states.t[i] = st.time.second + st.time.minute * 60.0
        states.x[i] = st.position[0]
        states.y[i] = st.position[1]
        states.z[i] = st.position[2]
        states.vx[i] = st.velocity[0]
        states.vy[i] = st.velocity[1]
        states.vz[i] = st.velocity[2]

    return states


def read_txt_file(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()

    states = OrbitStates.empty(len(lines))

    for i, line in enumerate(lines):
        try:
            (
                states.t[i],
                states.x[i],
                states.y[i],
                states.z[i],
                states.vx[i],
                states.vy[i],
                states.vz[i]
            ) = [float(t) for t in line.split()]
        except ValueError as e:
            raise ValueError(
                "I need {} to be a 7 column text file, with columns t, x, y, "
                "z, vx, vy, vz (Couldn't parse line {})"
                .format(filename, repr(line))
            ) from e

    return states


def read_ESA_Orbit_file(filename):
    '''
    Read orbit data from an orbit file supplied by ESA
    '''
    tree = ET.parse(filename)
    root = tree.getroot()
    data_block = root[1]
    numOSV = len(data_block[0])

    states = OrbitStates.empty(numOSV)

    for i, st in enumerate(data_block[0]):
        states.t[i] = datetime.datetime.strptime(
            st[1].text,
            '%Z=%Y-%m-%dT%H:%M:%S.%f'
        ).timestamp()
        states.x[i] = float(st[4].text)
        states.y[i] = float(st[5].text)
        states.z[i] = float(st[6].text)
        states.vx[i] = float(st[7].text)
        states.vy[i] = float(st[8].text)
        states.vz[i] = float(st[9].text)

    t = states.t
    t -= states.t[0]

    return states


def read_los_file(filepath):
    """
    Read incidence-heading information from a .rdr line of site file.
    """
    incidence, heading = [f.flatten() for f in gdal_open(filepath)]
    if incidence.shape != heading.shape:
        raise ValueError(
            "Malformed los file. Incidence shape {} and heading shape {} "
            "do not match!".format(incidence.shape, heading.shape)
        )
    return incidence, heading
