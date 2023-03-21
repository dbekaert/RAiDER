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
from RAiDER.logger import logger

wm = 'ERA5' if WM == 'ERA-5' else WM

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


def test_cube_timemean():
    """ Test the mean interpolation by computing cube delays at 1:30PM vs mean of 12 PM / 3PM for GMAO """
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERP_TIME")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    ## make the lat lon grid
    S, N, W, E = 34, 35, -117, -116
    date       = 20200130
    dct_hrs    = {'GMAO': [12, 15, '13:30:00'], 'ERA5': [13, 14, '13:30:00']}
    hr1, hr2, ti = dct_hrs[wm]

    grp = {
            'date_group': {'date_start': date},
            'weather_model': WM,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }


    ## run raider original for two exact weather model times
    for hr in [hr1, hr2]:
        grp['time_group'] =  {'time': f'{hr}:00:00'}
        ## generate the default template file and overwrite it with new parms
        cfg  = update_yaml(grp)

        ## run raider for the default date
        cmd  = f'raider.py {cfg}'
        proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        assert np.isclose(proc.returncode, 0)

    ## run interpolation in the middle of the two
    grp['time_group'] =  {'time': ti, 'interpolate_time': True}
    cfg  = update_yaml(grp)

    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)


    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{hr1}0000_ztd.nc')) as ds:
        da1_tot = ds['wet'] + ds['hydro']

    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{hr2}0000_ztd.nc')) as ds:
        da2_tot = ds['wet'] + ds['hydro']

    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{ti.replace(":", "")}_ztd.nc')) as ds:
        da_interp_tot = ds['wet'] + ds['hydro']

    da_mu = (da1_tot + da2_tot) / 2
    assert np.allclose(da_mu, da_interp_tot)


    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{WM}*')]
    os.remove('temp.yaml')

    return


def test_cube_weighting():
    """ Test the weighting by comparing a small crop with numpy directly """
    from datetime import datetime
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERP_TIME")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    ## make the lat lon grid
    S, N, W, E = 34, 35, -117, -116
    date       = 20200130
    dct_hrs    = {'GMAO': [12, 15, '12:05:00'], 'ERA5': [13, 14, '13:05:00']}
    hr1, hr2, ti = dct_hrs[WM]

    grp = {
            'date_group': {'date_start': date},
            'weather_model': WM,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }


    ## run raider original for two exact weather model times
    for hr in [hr1, hr2]:
        grp['time_group'] =  {'time': f'{hr}:00:00'}
        ## generate the default template file and overwrite it with new parms
        cfg  = update_yaml(grp)

        ## run raider for the default date
        cmd  = f'raider.py {cfg}'
        proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        assert np.isclose(proc.returncode, 0)

    ## run interpolation very near the first
    grp['time_group'] =  {'time': ti, 'interpolate_time': True}
    cfg  = update_yaml(grp)

    cmd  = f'raider.py {cfg}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)

    ## double check on weighting

    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{hr1}0000_ztd.nc')) as ds:
        da1_tot = ds['wet'] + ds['hydro']

    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{hr2}0000_ztd.nc')) as ds:
        da2_tot = ds['wet'] + ds['hydro']

    with xr.open_dataset(os.path.join(SCENARIO_DIR, f'{WM}_tropo_{date}T{ti.replace(":", "")}_ztd.nc')) as ds:
        da_interp_tot = ds['wet'] + ds['hydro']

    dt1 = datetime.strptime(f'{date}{hr1}', '%Y%m%d%H')
    dt2 = datetime.strptime(f'{date}{hr2}', '%Y%m%d%H')
    dt_ref = datetime.strptime(f'{date}{ti}', '%Y%m%d%H:%M:%S')

    wgts  = np.array([(dt_ref-dt1).seconds, (dt2-dt_ref).seconds])
    da1_crop = da1_tot.isel(z=0, y=slice(0,1), x=slice(0, 2))
    da2_crop = da2_tot.isel(z=0, y=slice(0,1), x=slice(0, 2))
    da_out_crop = da_interp_tot.isel(z=0, y=slice(0,1), x=slice(0,2))

    dat = np.vstack([da1_crop.data, da2_crop.data])

    logger.info ('Tstart: %s, Tend: %s, Tref: %s', dt1, dt2, dt_ref)
    logger.info ('Weights: %s', wgts)
    logger.info ('Data from two dates: %s', dat)
    logger.info ('Weighted mean: %s', da_out_crop.data)
    assert np.allclose(da_out_crop, np.average(dat, weights=1/wgts, axis=0))

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    [os.remove(f) for f in glob.glob(f'{WM}*')]
    os.remove('temp.yaml')

    return

