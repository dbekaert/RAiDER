'''
When update_rc_file is either True or False, the relevant API RC file should be
created if it doesn't exist.
'''
import os
from pathlib import Path
from platform import system
from RAiDER.models import credentials
from test import random_string


def test_cds():
    import cdsapi

    model_name = 'ERA5'

    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES[model_name])
    rc_path = rc_path.expanduser()
    rc_path.unlink(missing_ok=True)

    test_uid = random_string()
    test_key = random_string()

    # Test creation of .cdsapirc file in current dir
    credentials.check_api(model_name, test_uid, test_key, './', update_rc_file=False)
    assert rc_path.exists(), f'{rc_path} was not created'

    # Check the content
    cds_credentials = cdsapi.api.read_config(rc_path)
    uid, key = cds_credentials['key'].split(':')

    # Remove local API file
    rc_path.unlink()

    assert uid == test_uid, f'{rc_path}: UID was not written correctly'
    assert key == test_key, f'{rc_path}: KEY was not written correctly'


def test_ecmwf():
    import ecmwfapi

    model_name = 'HRES'

    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES[model_name])
    rc_path = rc_path.expanduser()
    rc_path.unlink(missing_ok=True)

    test_uid = random_string()
    test_key = random_string()

    # Test creation of .ecmwfapirc file
    credentials.check_api(model_name, test_uid, test_key, './', update_rc_file=False)
    assert rc_path.exists(), f'{rc_path} does not exist'

    # Get current ECMWF API RC file path
    old_rc_path = os.getenv("ECMWF_API_RC_FILE", ecmwfapi.api.DEFAULT_RCFILE_PATH)

    # Point ecmwfapi to current dir to avoid overwriting ~/.ecmwfapirc
    os.environ["ECMWF_API_RC_FILE"] = str(rc_path)
    key, _, uid = ecmwfapi.api.get_apikey_values()

    # Point ecmwfapi back to previous value and remove local API file
    os.environ["ECMWF_API_RC_FILE"] = old_rc_path
    rc_path.unlink()

    # Check if API is written correctly
    assert uid == test_uid, f'{rc_path}: UID was not written correctly'
    assert key == test_key, f'{rc_path}: KEY was not written correctly'


def test_netrc():
    import netrc

    model_name = 'GMAO'

    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES[model_name])
    rc_path = rc_path.expanduser()
    rc_path.unlink(missing_ok=True)

    test_uid = random_string()
    test_key = random_string()

    # Test creation of .netrc file
    credentials.check_api(model_name, test_uid, test_key, './', update_rc_file=False)
    assert os.path.exists(rc_path), f'{rc_path} does not exist'

    # Check the content
    host = 'urs.earthdata.nasa.gov'
    netrc_credentials = netrc.netrc(rc_path)
    uid, _, key = netrc_credentials.authenticators(host)

    # Remove local API file
    rc_path.unlink()

    assert uid == test_uid, f'{rc_path}: UID was not written correctly'
    assert key == test_key, f'{rc_path}: KEY was not written correctly'
