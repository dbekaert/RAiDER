import os
import shutil
import numpy as np
import pytest
import shutil
import subprocess
import xarray as xr
import rasterio as rio

from test import TEST_DIR


def test_GUNW():
    ## eventually to be implemented
    # home = os.path.expanduser('~')
    # netrc = os.path.join(home, '.netrc')
# 
    # ## make netrc
    # if not os.path.exists(netrc):
        # name, passw = os.getenv('URSname'), os.getenv('URSpass')
        # cmd = f'echo "machine urs.earthdata.nasa.gov login {name} password {passw}" > ~/.netrc'
        # subprocess.run(cmd.split())
# 
        # cmd = f'chmod 600 {netrc}'
        # subprocess.run(cmd.split())
# 
    SCENARIO_DIR = os.path.join(TEST_DIR, "GUNW")
    os.makedirs(SCENARIO_DIR, exist_ok=True)
    GUNW = 'S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc'
    orig_GUNW = os.path.join(TEST_DIR, GUNW)
    updated_GUNW = os.path.join(SCENARIO_DIR, GUNW)
    shutil.copy(orig_GUNW, updated_GUNW)


    # cmd  = f'wget https://grfn.asf.alaska.edu/door/download/{GUNW}'
    # proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    # assert np.isclose(proc.returncode, 0)

    cmd  = f'raider.py ++process calcDelaysGUNW {updated_GUNW} -m GMAO -o {SCENARIO_DIR}'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    ## check the CRS and affine are written correctly
    epsg      = 4326
    transform = (0.1, 0.0, -119.35, 0, -0.1, 35.05)
    group = 'science/grids/corrections/external/troposphere'
    for v in 'troposphereWet troposphereHydrostatic'.split():
        with rio.open(f'netcdf:{updated_GUNW}:{group}/{v}') as ds:
            assert np.isclose(ds.crs.to_epsg(), epsg), 'CRS incorrect'
            assert ds.transform.almost_equals(transform), 'Affine Transform incorrect'

    with xr.open_dataset(updated_GUNW, group=group) as ds:
        for v in 'troposphereWet troposphereHydrostatic'.split():
            da = ds[v]
            assert da.rio.transform().almost_equals(transform), 'Affine Transform incorrect'

        crs = rio.crs.CRS.from_wkt(ds['crs'].crs_wkt)
        assert np.isclose(crs.to_epsg(), epsg), 'CRS incorrect'
        

    # Clean up files
    shutil.rmtree(SCENARIO_DIR)
    os.remove('GUNW_20200130-20200124.yaml')
    return 
