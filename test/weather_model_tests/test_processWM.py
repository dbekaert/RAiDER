import operator
from functools import reduce

import numpy as np
import pytest
from numpy import nan

from RAiDER.processWM import getWMFilename


def test_getWMFilename():
    flag, f = getWMFilename(
        'ERA5', 
        datetime(2020,1,1,0,0,0), 
        'test_out_loc'
    )
    assert f == 
        os.path.join(
            'test_out_loc', 
            'ERA5_2020_01_01_00_00_00.nc'
        )


