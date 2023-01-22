import os
import pytest

import numpy as np
import xarray as xr


from test import TEST_DIR

from RAiDER.delayFcns import getInterpolators

SCENARIO1_DIR = os.path.join(TEST_DIR, "scenario_1", "golden_data")


@pytest.fixture
def wmdata():
    return xr.load_dataset(os.path.join(SCENARIO1_DIR, 'HRRR_tropo_20200101T120000_ztd.nc'))


def test_getInterpolators(wmdata):
    ds = wmdata
    tw, th = getInterpolators(ds, kind='pointwise')
    assert True # always pass unless an error is raised

def test_getInterpolators_2(wmdata):
    ds = wmdata
    ds['hydro'][0,0,0] = np.nan
    with pytest.raises(RuntimeError):
        getInterpolators(ds, kind='pointwise')
    
