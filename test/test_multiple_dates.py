import os
import pytest
import subprocess
import yaml
import numpy as np
from test import TEST_DIR


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
       'date_group': {'date_start': '20200103'},
       'time_group': {'time': '00:00:00'},
       'weather_model': 'ERA5',
      }
    
    params = {**params, **dct_group}
    dst = 'temp.yaml'
    
    with open(dst, 'w') as fh:
        yaml.dump(params, fh, default_flow_style=False)

    
    cmd  = f'raider.py temp.yaml'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)
    
    return dst
