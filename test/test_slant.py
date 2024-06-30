import glob

import numpy as np
import xarray as xr

from test import *

@pytest.mark.parametrize('weather_model_name', ['ERA5'])
def test_slant_proj(weather_model_name):
    SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_3")
    os.makedirs(SCENARIO_DIR, exist_ok=True)

    ## make the lat lon grid
    S, N, W, E = 33, 34, -118.25, -116.75
    date       = 20200130
    time       ='13:52:45'

    ## make the run config file
    grp = {
            'date_group': {'date_start': date},
            'height_group': {'height_levels': [0, 100, 500, 1000]},
            'time_group': {'time': time, 'interpolate_time': 'none'},
            'weather_model': weather_model_name,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'runtime_group': {'output_directory': SCENARIO_DIR,
                              'weather_model_directory': WM_DIR,
                              },
           'los_group' : {'ray_trace': False,
                          'orbit_file': os.path.join(ORB_DIR,
                                'S1B_OPER_AUX_POEORB_OPOD_20210317T025713_'\
                                'V20200129T225942_20200131T005942.EOF')
           }
        }

    ## generate the default run config file and overwrite it with new parms
    cfg  = update_yaml(grp, 'temp.yaml')

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER Failed.'

    gold = {'ERA5': [33.4, -117.8, 0, 2.333865144]}
    lat, lon, hgt, val = gold[weather_model_name]
    path_delays = os.path.join(SCENARIO_DIR,
                    make_delay_name(weather_model_name, date, time, 'std'))
    with xr.open_dataset(path_delays) as ds:
        delay = (ds['hydro'] + ds['wet']).sel(
            y=lat, x=lon, z=hgt, method='nearest').item()

    np.testing.assert_almost_equal(val, delay)

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{weather_model_name}*')]
    os.remove('temp.yaml')


@pytest.mark.parametrize('weather_model_name', ['ERA5'])
def test_ray_tracing(weather_model_name):
    SCENARIO_DIR = os.path.join(TEST_DIR, "scenario_3")
    os.makedirs(SCENARIO_DIR, exist_ok=True)

    ## make the lat lon grid
    S, N, W, E = 33, 34, -118.25, -117.25
    date       = 20200130
    time       ='13:52:45'

    ## make the run config file
    grp = {
            'date_group': {'date_start': date},
            'height_group': {'height_levels': [0, 100, 500, 1000]},
            'time_group': {'time': time, 'interpolate_time': 'none'},
            'weather_model': weather_model_name,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'runtime_group': {'output_directory': SCENARIO_DIR,
                              'weather_model_directory': WM_DIR,
                              },
           'los_group' : {'ray_trace': True,
                          'orbit_file': os.path.join(ORB_DIR,
                                'S1B_OPER_AUX_POEORB_OPOD_20210317T025713_'\
                                'V20200129T225942_20200131T005942.EOF')
           }
        }

    ## generate the default run config file and overwrite it with new parms
    cfg  = update_yaml(grp, 'temp.yaml')

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER Failed.'

    # model to lat/lon/correct value
    gold = {'ERA5': [33.4, -117.8, 0, 2.97711681]}
    lat, lon, hgt, val = gold[weather_model_name]

    path_delays = os.path.join(SCENARIO_DIR,
                    make_delay_name(weather_model_name, date, time, 'ray'))
    with xr.open_dataset(path_delays) as ds:
        delay = (ds['hydro'] + ds['wet']).sel(
            y=lat, x=lon, z=hgt, method='nearest').item()
    np.testing.assert_almost_equal(val, delay)

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{weather_model_name}*')]
    os.remove('temp.yaml')

