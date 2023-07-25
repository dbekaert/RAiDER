import os
import pytest

import numpy as np

from datetime import datetime
from test import TEST_DIR

from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.erai import ERAI


@pytest.mark.long
def test_era5():
    wm = ERA5()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_era5.nc'),
        datetime(2020, 1, 1, 0, 0, 0)
    )


@pytest.mark.long
def test_era5t():
    wm = ERA5T()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_era5t.nc'),
        datetime(2020, 1, 1, 0, 0, 0)
    )


@pytest.mark.long
def test_erai():
    wm = ERAI()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        os.path.join(TEST_DIR, 'test_geom', 'test_erai.nc'),
        datetime(2017, 1, 1, 0, 0, 0)
    )
