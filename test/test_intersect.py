import glob
import shutil

import pandas as pd
import rioxarray as xrr

from test import *


@pytest.mark.parametrize('wm', 'ERA5'.split())
def test_cube_intersect(wm):
    """ Test the intersection of lat/lon files with the DEM (model height levels?) """
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERSECT")
    os.makedirs(SCENARIO_DIR, exist_ok=True)

    ## make the lat lon grid
    S, N, W, E = 33, 34.5, -118.0, -117.0
    date       = 20200130
    time       ='13:52:45'
    f_lat, f_lon = makeLatLonGrid([S, N, W, E], 'LA', SCENARIO_DIR, 0.5)

    ## make the template file
    grp = {
            'date_group': {'date_start': date},
            'time_group': {'time': time, 'interpolate_time': False},
            'weather_model': wm,
            'aoi_group': {'lat_file': f_lat, 'lon_file': f_lon},
            'runtime_group': {'output_directory': SCENARIO_DIR,
                              'weather_model_directory': WM_DIR,
                              },
        }

    ## generate the default template file and overwrite it with new parms
    cfg  = update_yaml(grp, 'temp.yaml')

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER Failed.'

    ## hard code what it should be and check it matches
    gold = {'ERA5': 2.2906463, 'GMAO': np.nan, 'HRRR': np.nan}

    path_delays = os.path.join(SCENARIO_DIR, f'{wm}_hydro_{date}T{time.replace(":", "")}_ztd.tiff')
    da  = xrr.open_rasterio(path_delays, band_as_variable=True)['band_1']
    hyd = da.sel(x=-117.8, y=33.4, method='nearest').item()
    np.testing.assert_almost_equal(gold[wm], hyd)

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{wm}*')]
    os.remove('temp.yaml')

    return


@pytest.mark.parametrize('wm', 'ERA5'.split())
def test_gnss_intersect(wm):
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERSECT")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    gnss_file = os.path.join(TEST_DIR, 'scenario_2', 'stations2.csv')
    date       = 20200130
    time       ='13:52:45'

    ## make the template file
    grp = {
            'date_group': {'date_start': date},
            'time_group': {'time': time, 'interpolate_time': False},
            'weather_model': wm,
            'aoi_group': {'station_file': gnss_file},
            'runtime_group': {'output_directory': SCENARIO_DIR,
                              'weather_model_directory': WM_DIR,
                            }
        }

    ## generate the default template file and overwrite it with new parms
    cfg  = update_yaml(grp)

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert proc.returncode == 0, 'RAiDER Failed.'

    gold = {'ERA5': 2.3451457, 'GMAO': np.nan, 'HRRR': np.nan}

    df = pd.read_csv(os.path.join(SCENARIO_DIR, f'{wm}_Delay_{date}T{time.replace(":", "")}.csv'))

    id = 'TORP'
    td = df.set_index('ID').loc[id, 'totalDelay']
    np.testing.assert_almost_equal(gold[wm], td.item())

    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{wm}*')]
    os.remove('temp.yaml')

    return
