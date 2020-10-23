import datetime
import os
import pytest


from RAiDER.processWM import getWMFilename


def test_getWMFilename():
    f = getWMFilename(
        'ERA5',
        datetime.datetime(2020, 1, 1, 0, 0, 0),
        'test_out_loc'
    )
    assert f[0]
    assert f[1] == os.path.join(
        'test_out_loc',
        'ERA5_2020_01_01_T00_00_00.nc'
    )
