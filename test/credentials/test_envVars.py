'''
Environment variables specific to each model are accepted iff uid and key
arguments are None.
'''

from pathlib import Path
from platform import system

import pytest
from test import random_string

from RAiDER.models import credentials


@pytest.mark.parametrize(
    'model_name,template,env_var_name_uid,env_var_name_key',
    [
        (
            'ERA5', (
                'url: https://cds.climate.copernicus.eu/api\n'
                'key: {uid}:{key}\n'
            ),
            'RAIDER_ECMWF_ERA5_UID',
            'RAIDER_ECMWF_ERA5_API_KEY'
        ),
        (
            'ERA5T', (
                'url: https://cds.climate.copernicus.eu/api\n'
                'key: {uid}:{key}\n'
            ),
            'RAIDER_ECMWF_ERA5_UID',
            'RAIDER_ECMWF_ERA5_API_KEY'
        ),
        (
            'HRES', (
                '{{\n'
                '    "url"   : "https://api.ecmwf.int/v1",\n'
                '    "key"   : "{key}",\n'
                '    "email" : "{uid}"\n'
                '}}\n'
            ),
            'RAIDER_HRES_EMAIL',
            'RAIDER_HRES_API_KEY'
        ),
        (
            # Simulate a .netrc file with multiple sets of credentials.
            # The ones not for urs.earthdata.nasa.gov should NOT be touched.
            # Indentation is done with TABS, as that is what the netrc package
            # generates.
            'GMAO', (
                'machine example.com\n'
                '	login johndoe\n'
                '	password hunter2\n'
                'machine urs.earthdata.nasa.gov\n'
                '	login {uid}\n'
                '	password {key}\n'
                'machine 127.0.0.1\n'
                '	login bobsmith\n'
                '	password dolphins\n'
            ),
            'EARTHDATA_USERNAME',
            'EARTHDATA_PASSWORD'
        ),
        (
            'MERRA2', (
                'machine example.com\n'
                '	login johndoe\n'
                '	password hunter2\n'
                'machine urs.earthdata.nasa.gov\n'
                '	login {uid}\n'
                '	password {key}\n'
                'machine 127.0.0.1\n'
                '	login bobsmith\n'
                '	password dolphins\n'
            ),
            'EARTHDATA_USERNAME',
            'EARTHDATA_PASSWORD'
        ),
    ]
)
def test_envVars(
    monkeypatch,
    model_name,
    template,
    env_var_name_uid,
    env_var_name_key
):
    hidden_ext = '_' if system() == "Windows" else '.'
    rc_filename = credentials.RC_FILENAMES[model_name]
    if rc_filename is None:
        return
    rc_path = Path('./') / (hidden_ext + rc_filename)
    rc_path = rc_path.expanduser()
    rc_path.unlink(missing_ok=True)

    # Give the rc file mock contents
    rc_path.write_text(template.format(uid=random_string(), key=random_string()))

    test_uid = random_string()
    test_key = random_string()

    with monkeypatch.context() as mp:
        mp.setenv(env_var_name_uid, test_uid)
        mp.setenv(env_var_name_key, test_key)
        credentials.check_api(model_name, None, None, './', update_rc_file=True)

    expected_content = template.format(uid=test_uid, key=test_key)
    actual_content = rc_path.read_text()
    rc_path.unlink()

    assert expected_content == actual_content, f'{rc_path} was not created correctly'
