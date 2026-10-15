from RAiDER.cli.raider import calcDelays
import pytest
import os

import numpy as np
import xarray as xr

from test import (
    ORB_DIR, make_delay_name, pushd
)
from RAiDER.utilFcns import write_yaml


@pytest.mark.parametrize('weather_model_name', ['ERA5'])
def test_slant_proj(weather_model_name, tmp_path, weather_model_dir):
    SCENARIO_DIR = tmp_path / "scenario_3"

    # make the lat lon grid
    S, N, W, E = 33, 34, -118.25, -116.75
    date = 20200130
    time = '13:52:45'

    # make the run config file
    grp = {
        'date_group': {'date_start': date},
        'height_group': {'height_levels': [0, 100, 500, 1000]},
        'time_group': {'time': time, 'interpolate_time': 'none'},
        'weather_model': weather_model_name,
        'aoi_group': {'bounding_box': [S, N, W, E]},
        'runtime_group': {'output_directory': SCENARIO_DIR,
                          'weather_model_directory': weather_model_dir,
                          },
        'los_group': {'ray_trace': False,
                      'orbit_file': os.path.join(ORB_DIR,
                                                 'S1B_OPER_AUX_POEORB_OPOD_20210317T025713_'\
                                                 'V20200129T225942_20200131T005942.EOF')
                      }
    }

    # generate the default run config file and overwrite it with new parms
    cfg = write_yaml(grp, tmp_path / 'temp.yaml')

    # run raider and intersect
    with pushd(tmp_path):
        calcDelays([str(cfg)])

    gold = {'ERA5': [33.4, -117.8, 0, 2.3324788251164725]}
    lat, lon, hgt, val = gold[weather_model_name]
    path_delays = os.path.join(SCENARIO_DIR,
                               make_delay_name(weather_model_name, date, time, 'std'))
    with xr.open_dataset(path_delays) as ds:
        delay = (ds['hydro'] + ds['wet']).sel(
            y=lat, x=lon, z=hgt, method='nearest').item()

    np.testing.assert_almost_equal(val, delay)


@pytest.mark.parametrize('weather_model_name', ['ERA5'])
def test_ray_tracing(weather_model_name, tmp_path, weather_model_dir):
    SCENARIO_DIR = tmp_path / "scenario_3"

    # make the lat lon grid
    S, N, W, E = 33, 34, -118.25, -117.25
    date = 20200130
    time = '13:52:45'

    # make the run config file
    grp = {
        'date_group': {'date_start': date},
        'height_group': {'height_levels': [0, 100, 500, 1000]},
        'time_group': {'time': time, 'interpolate_time': 'none'},
        'weather_model': weather_model_name,
        'aoi_group': {'bounding_box': [S, N, W, E]},
        'runtime_group': {'output_directory': SCENARIO_DIR,
                          'weather_model_directory': weather_model_dir,
                          },
        'los_group': {'ray_trace': True,
                      'orbit_file': os.path.join(ORB_DIR,
                                                 'S1B_OPER_AUX_POEORB_OPOD_20210317T025713_'\
                                                 'V20200129T225942_20200131T005942.EOF')
                      }
    }

    # generate the default run config file and overwrite it with new parms
    cfg = write_yaml(grp, tmp_path / 'temp.yaml')

    # run raider and intersect
    with pushd(tmp_path):
        calcDelays([str(cfg)])

    # model to lat/lon/correct value
    gold = {'ERA5': [33.4, -117.8, 0, 2.9756965061161926]}
    lat, lon, hgt, val = gold[weather_model_name]

    path_delays = os.path.join(SCENARIO_DIR,
                               make_delay_name(weather_model_name, date, time, 'ray'))
    with xr.open_dataset(path_delays) as ds:
        delay = (ds['hydro'] + ds['wet']).sel(
            y=lat, x=lon, z=hgt, method='nearest').item()
    np.testing.assert_almost_equal(val, delay)
