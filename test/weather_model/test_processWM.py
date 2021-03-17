import os
import pytest

import numpy as np

from test import TEST_DIR

from RAiDER.models.era5 import ERA5
from RAiDER.processWM import checkContainment


SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")


@pytest.fixture
def getWM():
    wm = ERA5()
    wm.files = [
        os.path.join(
            SCENARIO_DIR, 'weather_files/ERA-5_2020_01_03_T23_00_00.nc'
        )
    ]
    return wm


def test_checkContainment(getWM):
    wm = getWM
    outLats = np.linspace(10, 20)
    outLons = -100 * np.ones(outLats.shape)

    containment = checkContainment(wm, outLats, outLons)
    assert(~containment)


def test_checkContainment2(getWM):
    wm = getWM
    outLats = np.linspace(17, 18)
    outLons = -100 * np.ones(outLats.shape)

    containment = checkContainment(wm, outLats, outLons)
    assert(containment)
