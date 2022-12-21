import os
import glob
import pytest
import subprocess
import shutil
import yaml
import numpy as np
from test import TEST_DIR

## ToDo check where outputs are created/stored/found and clean up
def test_dates():
    ## make a default template file
    cmd  = f'raider.py -g'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## write a new file
    with open('./raider.yaml', 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f'Something is wrong with the yaml file {example_yaml}')
    

    dct_group = {
       'aoi_group': {'bounding_box': [28, 39, -123, -112]},
       'date_group': {'date_list': ['20200124', '20200130']},
       'time_group': {'time': '00:00:00'},
       'weather_model': 'ERA5',
      }
    
    params = {**params, **dct_group}
    dst = 'temp.yaml'
    
    with open(dst, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)

    
    ## run raider on new file (two dates)
    cmd  = f'raider.py temp.yaml'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## check that four files (2x date) were created
    n_files = glob.glob('weather_files/*.nc')
    assert np.isclose(n_files, 4), 'Incorrect number of files produced'

    ## clean up
    # shutil.remove('./weather_files')?
    
    return dst
