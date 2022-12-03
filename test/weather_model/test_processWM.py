import os
import pytest

import numpy as np

from test import TEST_DIR

from RAiDER.models.era5 import ERA5


SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")


@pytest.fixture
def getWM():
    """The bounds of the ERA model are:

    BoundingBox(left=-103.7750015258789,
                bottom=15.225000381469727,
                right=-99.2750015258789,
                top=18.725000381469727)
    """
    wm = ERA5()
    wm.files = [
        os.path.join(
            SCENARIO_DIR, 'weather_files/ERA-5_2020_01_03_T23_00_00.nc'
        )
    ]
    return wm


def test_checkContainment(getWM):
    if os.path.exists(getWM.files[0]):
        wm = getWM
        ll_bounds = (10, 20, -100, -100)
        # outLats = np.linspace(10, 20)
        # outLons = -100 * np.ones(outLats.shape)

        containment = wm.checkContainment(ll_bounds)
        assert(~containment)


def test_checkContainment2(getWM):
    if os.path.exists(getWM.files[0]):
        wm = getWM
        # outLats = np.linspace(17, 18)
        # outLons = -100 * np.ones(outLats.shape)
        ll_bounds = (17, 18, -100, -100)

        containment = wm.checkContainment(ll_bounds)
        assert(containment)
