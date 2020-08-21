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
