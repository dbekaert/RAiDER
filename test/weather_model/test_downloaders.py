import os
import pytest

import numpy as np

from datetime import datetime
from test import TEST_DIR, DATA_DIR, pushd

from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.erai import ERAI


def test_era5():
    wm = ERA5()
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_era5.nc'),
        np.array([10, 10.1, 10.2]),
        np.array([-72, -72, -72]),
        datetime(2020, 1, 1, 0, 0, 0)
    )


def test_era5t():
    wm = ERA5T()
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_era5t.nc'),
        np.array([10, 10.1, 10.2]),
        np.array([-72, -72, -72]),
        datetime(2020, 1, 1, 0, 0, 0)
    )


@pytest.mark.xfail(reason='ECMWF API is not working for some reason, need to revisit')
def test_erai():
    wm = ERAI()
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_erai.nc'),
        np.array([10, 10.1, 10.2]),
        np.array([-72, -72, -72]),
        datetime(2017, 1, 1, 0, 0, 0)
    )
