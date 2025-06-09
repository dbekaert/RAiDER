from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from RAiDER.models.era5 import ERA5
from RAiDER.models.era5t import ERA5T
from RAiDER.models.erai import ERAI
from test import TEST_DIR


@pytest.mark.long
def test_era5() -> None:
    wm = ERA5()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_era5.nc',
        datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=timezone(offset=timedelta()))
    )


@pytest.mark.long
def test_era5t() -> None:
    wm = ERA5T()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_era5t.nc',
        datetime(2020, 1, 1, 0, 0, 0).replace(tzinfo=timezone(offset=timedelta()))
    )


@pytest.mark.long
def test_erai() -> None:
    wm = ERAI()
    wm.set_latlon_bounds(np.array([10, 10.2, -72, -72]))
    wm.fetch(
        TEST_DIR / 'test_geom/test_erai.nc',
        datetime(2017, 1, 1, 0, 0, 0).replace(tzinfo=timezone(offset=timedelta()))
    )
