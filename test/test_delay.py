import os
import pytest

import numpy as np

from test import DATA_DIR, pushd

from RAiDER.delay import (
    checkQueryPntsFile
)


#@pytest.fixture
#def llsimple():
#    lats = (10, 12)
#    lons = (-72, -74)
#    return lats, lons

def test_cqpf1():
    assert checkQueryPntsFile('does_not_exist.h5', None)

