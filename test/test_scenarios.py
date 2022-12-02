import os
import pytest
import subprocess

from test import TEST_DIR

import numpy as np
import xarray as xr


@pytest.mark.long
def test_scenario_1():
    SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")
    test_path = os.path.join(SCENARIO_DIR, 'raider_example_1.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True,)
    assert process.returncode == 0

    new_data = xr.load_dataset('HRRR_tropo_20200101T120000_ztd.nc')
    golden_data = xr.load_dataset(os.path.join(SCENARIO_DIR, 'HRRR_tropo_20200101T120000_ztd.nc'))

    assert np.allclose(golden_data['wet'], new_data['wet'])
    assert np.allclose(golden_data['hydro'], new_data['hydro'])


    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])


#TODO: Next release include GNSS station test
# @pytest.mark.long
# def test_scenario_2():
#     test_path = os.path.join(TEST_DIR, "scenario_2", 'raider_example_2.yaml')
#     process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True)
#     assert process.returncode == 0

#     # Clean up files
#     subprocess.run(['rm', '-f', './HRRR*'])
#     subprocess.run(['rm', '-rf', './weather_files'])


@pytest.mark.long
def test_scenario_3():
    SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_3")

    test_path = os.path.join(SCENARIO_DIR, 'raider_example_3.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True)
    assert process.returncode == 0

    new_data = xr.load_dataset('HRRR_tropo_20181113T230000_std.nc')
    golden_data = xr.load_dataset(os.path.join(SCENARIO_DIR, 'HRRR_tropo_20181113T230000_std.nc'))

    assert np.allclose(golden_data['wet'], new_data['wet'])
    assert np.allclose(golden_data['hydro'], new_data['hydro'])

    # Clean up files
    subprocess.run(['rm', '-f', './HRRR*'])
    subprocess.run(['rm', '-rf', './weather_files'])
