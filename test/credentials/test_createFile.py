'''
When update_rc_file is either True or False, the relevant API RC file should be
created if it doesn't exist.
'''
from typing import Tuple

import pytest
import os
from pathlib import Path
from platform import system
from RAiDER.models import credentials
from test import random_string


def get_creds_cds(rc_path: Path) -> Tuple[str, str]:
    import cdsapi
    cds_credentials = cdsapi.api.read_config(rc_path)
    uid, key = cds_credentials['key'].split(':')
    return uid, key


def get_creds_ecmwf(rc_path: Path) -> Tuple[str, str]:
    import ecmwfapi
    # Get current ECMWF API RC file path
    old_rc_path = os.getenv("ECMWF_API_RC_FILE", ecmwfapi.api.DEFAULT_RCFILE_PATH)

    # Point ecmwfapi to current dir to avoid overwriting ~/.ecmwfapirc
    os.environ["ECMWF_API_RC_FILE"] = str(rc_path)
    key, _, uid = ecmwfapi.api.get_apikey_values()

    # Point ecmwfapi back to previous value and remove local API file
    os.environ["ECMWF_API_RC_FILE"] = old_rc_path
    return uid, key


def get_creds_netrc(rc_path: Path) -> Tuple[str, str]:
    import netrc
    host = 'urs.earthdata.nasa.gov'
    netrc_credentials = netrc.netrc(rc_path)
    uid, _, key = netrc_credentials.authenticators(host)
    return uid, key


@pytest.mark.parametrize(
    'model_name,get_creds',
    (
        ('ERA5', get_creds_cds),
        ('ERA5T', get_creds_cds),
        ('HRES', get_creds_ecmwf),
        ('GMAO', get_creds_netrc),
        ('MERRA2', get_creds_netrc)
    )
)
def test_createFile(model_name, get_creds):
    # Get the rc file's path
    hidden_ext = '_' if system() == "Windows" else '.'
    rc_filename = credentials.RC_FILENAMES[model_name]
    if rc_filename is None:
        return
    rc_path = Path('./') / (hidden_ext + rc_filename)
    rc_path = rc_path.expanduser()
    rc_path.unlink(missing_ok=True)

    test_uid = random_string()
    test_key = random_string()

    # Test creation of the rc file
    credentials.check_api(model_name, test_uid, test_key, './', update_rc_file=False)
    assert rc_path.exists(), f'{rc_path} does not exist'

    # Check if API is written correctly
    uid, key = get_creds(rc_path)
    rc_path.unlink()
    assert uid == test_uid, f'{rc_path}: UID was not written correctly'
    assert key == test_key, f'{rc_path}: KEY was not written correctly'