from test import TEST_DIR

import numpy as np
import xarray as xr
from RAiDER.cli.raider import calcDelays

def test_scenario_1(data_for_hrrr_ztd, mocker):
    SCENARIO_DIR = TEST_DIR / "scenario_1"
    test_path = SCENARIO_DIR / 'raider_example_1.yaml'
    mocker.patch('RAiDER.processWM.prepareWeatherModel',
                 side_effect=[str(data_for_hrrr_ztd)])
    calcDelays([str(test_path)])

    new_data  = xr.load_dataset(SCENARIO_DIR / 'HRRR_tropo_20200101T120000_ztd.nc')
    new_data1 = new_data.sel(x=-91.84, y=36.84, z=0, method='nearest')
    golden_data = 2.2622863, 0.0361021 # hydro|wet

    np.testing.assert_almost_equal(golden_data[0], new_data1['hydro'].data)
    np.testing.assert_almost_equal(golden_data[1], new_data1['wet'].data)
