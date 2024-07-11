import pytest
import glob
import shutil
import os
import subprocess
import numpy as np
import xarray as xr


from test import (
    WM, TEST_DIR, update_yaml
)

from RAiDER.logger import logger

wm = 'ERA5' if WM == 'ERA-5' else WM


@pytest.mark.long
def test_cube_timemean():
    """ Test the mean interpolation by computing cube delays at 1:30PM vs mean of 12 PM / 3PM for GMAO """
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERP_TIME")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    ## make the lat lon grid
    S, N, W, E = 34, 35, -117, -116
    date       = 20200130
    dct_hrs    = {'GMAO': [12, 15, '13:30:00'], 'MERRA2': [12, 15, '13:30:00'], 'ERA5': [13, 14, '13:30:00']}
    hr1, hr2, ti = dct_hrs[wm]

    grp = {
            'date_group': {'date_start': date},
            'weather_model': WM,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'time_group': {'interpolate_time': 'none'},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }


    ## run raider without interpolation for two exact weather model times
    for hr in [hr1, hr2]:
        grp['time_group'].update({'time': f'{hr}:00:00'})
        ## generate the default run config file and overwrite it with new parms
        cfg  = update_yaml(grp)

        ## run raider for the default date
        cmd  = f'raider.py {cfg}'
        proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        assert np.isclose(proc.returncode, 0)

    ## run interpolation in the middle of the two
    grp['time_group'] =  {'time': ti, 'interpolate_time': 'center_time'}
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


@pytest.mark.long
def test_cube_weighting():
    """ Test the weighting by comparing a small crop with numpy directly """
    from datetime import datetime
    SCENARIO_DIR = os.path.join(TEST_DIR, "INTERP_TIME")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    ## make the lat lon grid
    S, N, W, E = 34, 35, -117, -116
    date       = 20200130
    dct_hrs    = {'GMAO': [12, 15, '12:05:00'], 'MERRA2': [12, 15, '13:30:00'], 'ERA5': [13, 14, '13:05:00']}
    hr1, hr2, ti = dct_hrs[WM]

    grp = {
            'date_group': {'date_start': date},
            'weather_model': WM,
            'aoi_group': {'bounding_box': [S, N, W, E]},
            'time_group': {'interpolate_time': 'none'},
            'runtime_group': {'output_directory': SCENARIO_DIR},
        }


    ## run raider without interpolation for two exact weather model times
    for hr in [hr1, hr2]:
        grp['time_group'].update({'time': f'{hr}:00:00'})
        ## generate the default run config file and overwrite it with new parms
        cfg  = update_yaml(grp)

        ## run raider for the default date
        cmd  = f'raider.py {cfg}'
        proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
        assert np.isclose(proc.returncode, 0)

    ## run interpolation very near the first
    grp['time_group'] =  {'time': ti, 'interpolate_time': 'center_time'}
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
