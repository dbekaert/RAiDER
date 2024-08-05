import os
import subprocess
import shutil
import glob

from test import TEST_DIR, WM, update_yaml, pushd

import numpy as np
import xarray as xr
from RAiDER.cli.raider import calcDelays


def test_scenario_1(tmp_path, data_for_hrrr_ztd, mocker):
    with pushd(tmp_path):
        dct_group = {
            "aoi_group": {"bounding_box": [36, 37, -92, -91]},
            "date_group": {"date_start": "20200101"},
            "time_group": {"time": "12:00:00", "interpolate_time": "none"},
            "weather_model": "HRRR",
            "height_group": {"height_levels": [0, 50, 100, 500, 1000]},
            "look_dir": "right",
            "runtime_group": {"output_directory": "test/scenario_1"},
        }

        cfg = update_yaml(dct_group, os.path.join(tmp_path, "temp.yaml"))

        SCENARIO_DIR = os.path.join(tmp_path, TEST_DIR, "scenario_1")
        mocker.patch(
            "RAiDER.processWM.prepareWeatherModel", side_effect=[str(data_for_hrrr_ztd)]
        )
        calcDelays([os.path.join(tmp_path, "temp.yaml")])

        new_data = xr.load_dataset(
            os.path.join(
                tmp_path, "test", "scenario_1", "HRRR_tropo_20200101T120000_ztd.nc"
            )
        )
        new_data1 = new_data.sel(x=-91.84, y=36.84, z=0, method="nearest")
        golden_data = 2.2622863, 0.0361021  # hydro|wet

        np.testing.assert_almost_equal(golden_data[0], new_data1["hydro"].data)
        np.testing.assert_almost_equal(golden_data[1], new_data1["wet"].data)
