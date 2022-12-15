import os
import shutil
import numpy as np
import pytest
import shutil
import subprocess

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
    os.chdir(SCENARIO_DIR)
    GUNW = 'S1-GUNW-D-R-071-tops-20200130_20200124-135156-34956N_32979N-PP-913f-v2_0_4.nc'
    shutil.copy(f'../{GUNW}', os.getcwd())


    # cmd  = f'wget https://grfn.asf.alaska.edu/door/download/{GUNW}'
    # proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    # assert np.isclose(proc.returncode, 0)

    cmd  = f'raider.py ++process calcDelaysGUNW {GUNW} -m ERA5'
    proc = subprocess.run(cmd.split(), stdout=subprocess.PIPE, universal_newlines=True)
    assert np.isclose(proc.returncode, 0)

    # Clean up files
    os.chdir('..')
    # shutil.rmtree(SCENARIO_DIR)

