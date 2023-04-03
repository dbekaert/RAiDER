import os
import glob
import pytest
import subprocess
import shutil
import yaml
import numpy as np
from test import TEST_DIR, WM


def test_datelist():
    SCENARIO_DIR = os.path.join(TEST_DIR, 'datelist')

    if os.path.exists(SCENARIO_DIR):
        shutil.rmtree(SCENARIO_DIR) 
    os.makedirs(SCENARIO_DIR, exist_ok=False)

    dates = ['20200124', '20200130']

    dct_group = {
       'aoi_group': {'bounding_box': [28, 39, -123, -112]},
       'date_group': {'date_list': dates},
       'time_group': {'time': '00:00:00'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': os.path.join(SCENARIO_DIR, 'weather_files')
            }
      }

    params = dct_group
    dst = os.path.join(SCENARIO_DIR, 'temp.yaml')

    with open(dst, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)


    ## run raider on new file (two dates)
    cmd  = f'raider.py {dst}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## check that four files (2x date) were created
    n_files = len(glob.glob(os.path.join(SCENARIO_DIR, 'weather_files/*.nc')))
    n_dates = len(dates)
    assert np.isclose(n_files, n_dates*2), 'Incorrect number of files produced'

    ## clean up
    shutil.rmtree(SCENARIO_DIR)

    return dst


def test_datestep():
    SCENARIO_DIR = os.path.join(TEST_DIR, 'datelist')
    os.makedirs(SCENARIO_DIR, exist_ok=False)
    st, en, step = '20200124', '20200130', 3
    n_dates      = 3

    dct_group = {
       'aoi_group': {'bounding_box': [28, 39, -123, -112]},
       'date_group': {'date_start': st, 'date_end': en, 'date_step': step},
       'time_group': {'time': '00:00:00'},
       'weather_model': WM,
       'runtime_group': {
            'output_directory': SCENARIO_DIR,
            'weather_model_directory': os.path.join(SCENARIO_DIR, 'weather_files')
            }
      }

    params = dct_group
    dst = os.path.join(SCENARIO_DIR, 'temp.yaml')

    with open(dst, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)


    ## run raider on new file (two dates)
    cmd  = f'raider.py {dst}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## check that four files (2x date) were created
    n_files = len(glob.glob(os.path.join(SCENARIO_DIR, 'weather_files/*.nc')))
    assert np.isclose(n_files, n_dates*2), 'Incorrect number of files produced'

    ## clean up
    shutil.rmtree(SCENARIO_DIR)
