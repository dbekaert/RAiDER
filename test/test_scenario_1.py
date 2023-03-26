import os
import pytest
import subprocess
import shutil
import glob

from test import TEST_DIR

import numpy as np
import xarray as xr


def test_scenario_1():
    SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_1")
    test_path = os.path.join(SCENARIO_DIR, 'raider_example_1.yaml')
    process = subprocess.run(['raider.py', test_path],stdout=subprocess.PIPE, universal_newlines=True,)
    assert process.returncode == 0

    new_data = xr.load_dataset(os.path.join(SCENARIO_DIR, 'HRRR_tropo_20200101T120000_ztd.nc'))
    new_data1= new_data.isel(y=10, x=10, z=0)
    
    golden_data = 3.36171181, 0.03765481 # hydro|wet

    
    assert np.isclose(new_data1['hydro'].data, golden_data[0])
    assert np.isclose(new_data1['wet'].data, golden_data[1])


    # Clean up files
    for f in glob.glob(os.path.join(SCENARIO_DIR, 'HRRR*')):
        os.remove(f)
    shutil.rmtree(os.path.join(SCENARIO_DIR, 'weather_files'))
