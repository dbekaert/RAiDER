import pytest
import os

from datetime import datetime

from RAiDER.processWM import getWMFilename


def test_getWMFilename():
    f = getWMFilename(
        'ERA5',
        datetime(2020, 1, 1, 0, 0, 0),
        'test_out_loc'
    )
    assert f == os.path.join(
        'test_out_loc',
        'ERA5_2020_01_01_T00_00_00.nc'
    )
