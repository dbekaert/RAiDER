import glob
import os
import pytest
import shutil
import subprocess
import yaml

import numpy as np
import pandas as pd
import xarray as xr

from test import TEST_DIR, WM

import RAiDER

wm = 'ERA-5' if WM == 'ERA5' else WM


def makeLatLonGrid(bbox, reg, out_dir, spacing=0.1):
    """ Make lat lons at a specified spacing """
    S, N, W, E = bbox
    lat_st, lat_en = S, N
    lon_st, lon_en = W, E

    lats = np.arange(lat_st, lat_en, spacing)
    lons = np.arange(lon_st, lon_en, spacing)
    Lat, Lon = np.meshgrid(lats, lons)
    da_lat = xr.DataArray(Lat.T, name='data', coords={'lon': lons, 'lat': lats}, dims='lat lon'.split())
    da_lon = xr.DataArray(Lon.T, name='data', coords={'lon': lons, 'lat': lats}, dims='lat lon'.split())

    dst_lat = os.path.join(out_dir, f'lat_{reg}.nc')
    dst_lon = os.path.join(out_dir, f'lon_{reg}.nc')
    da_lat.to_netcdf(dst_lat)
    da_lon.to_netcdf(dst_lon)

    return dst_lat, dst_lon


def update_yaml(dct_cfg:dict, dst:str='temp.yaml'):
    """ Write a new yaml file from a dictionary.

    Updates parameters in the default 'raider.yaml' file.
    Each key:value pair will in 'dct_cfg' will overwrite that in the default
    """

    template_file = os.path.join(
                    os.path.dirname(RAiDER.__file__), 'cli', 'raider.yaml')

    with open(template_file, 'r') as f:
        try:
            params = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise ValueError(f'Something is wrong with the yaml file {template_file}')

    params = {**params, **dct_cfg}

    with open(dst, 'w') as fh:
        yaml.safe_dump(params, fh,  default_flow_style=False)

    return dst


def test_cube_intersect():
    """ Test the intersection of lat/lon files with the DEM (model height levels?) """
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERSECT")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    ## make the lat lon grid
    S, N, W, E = 34, 35, -117, -116
    date       = 20200130
    time       ='12:00:00'
    f_lat, f_lon = makeLatLonGrid([S, N, W, E], 'LA', SCENARIO_DIR, 0.1)

    ## make the template file
    grp = {
            'date_group': {'date_start': date},
            'time_group': {'time': time},
            'weather_model': WM,
            'aoi_group': {'lat_file': f_lat, 'lon_file': f_lon},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }

    ## generate the default template file and overwrite it with new parms
    cfg  = update_yaml(grp)

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## hard code what it should be and check it matches
    WM_file = os.path.join(SCENARIO_DIR, f'{WM}_hydro_{date}T{time.replace(":", "")}_ztd.tiff')
    da      = xr.open_dataset(WM_file)['band_data']
    gold    = {'GMAO': 2.0541468, 'ERA5': 2.0696816}#, 'HRRR': 3.0972726}
    assert np.isclose(da.mean().round(6), gold[WM])

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{WM}*')]
    os.remove('temp.yaml')

    return


def test_gnss_intersect():
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERSECT")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    gnss_file = os.path.join(TEST_DIR, 'scenario_2', 'stations.csv')
    date       = 20200130
    time       ='12:00:00'

#     ## make the template file
    grp = {
            'date_group': {'date_start': date},
            'time_group': {'time': time},
            'weather_model': WM,
            'aoi_group': {'station_file': gnss_file},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }

    ## generate the default template file and overwrite it with new parms
    cfg  = update_yaml(grp)

    ## run raider and intersect
    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    gold = {'GMAO': 2.365131, 'ERA5': 2.39535}#, 'HRRR': 3.435141}
    df = pd.read_csv(os.path.join(SCENARIO_DIR, f'{WM}_Delay_{date}T{time.replace(":", "")}.csv'))
    td = df['totalDelay'].mean().round(6)
    assert np.allclose(gold[WM], td)

    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{WM}*')]
    os.remove('temp.yaml')

    return
