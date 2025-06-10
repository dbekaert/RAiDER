'''
When update_rc_file is True, the RC file should be:
- updated if it already exists,
- created if it doesn't,
- and for .netrc files, it should ONLY update the set of credentials related to
  the given weather model's API URL.
'''
from pathlib import Path
from platform import system

import pytest
from test import random_string

from RAiDER.models import credentials


@pytest.mark.parametrize(
    'model_name,template',
    [
        (
            'ERA5', (
                'url: https://cds.climate.copernicus.eu/api\n'
                'key: {uid}:{key}\n'
            )
        ),
        (
            'ERA5T', (
                'url: https://cds.climate.copernicus.eu/api\n'
                'key: {uid}:{key}\n'
            )
        ),
        (
            'HRES', (
                '{{\n'
                '    "url"   : "https://api.ecmwf.int/v1",\n'
                '    "key"   : "{key}",\n'
                '    "email" : "{uid}"\n'
                '}}\n'
            )
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
            )
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
            )
        ),
    ]
)
def test_updateTrue(model_name, template) -> None:
    # Get the rc file's path
    hidden_ext = '_' if system() == "Windows" else '.'
    rc_filename = credentials.RC_FILENAMES[model_name]
    if rc_filename is None:
        return
    rc_path = Path('./') / (hidden_ext + rc_filename)

    # Give the rc file mock contents
    rc_path.write_text(template.format(uid=random_string(), key=random_string()))

    # Use check_api to update the rc file
    test_uid = random_string()
    test_key = random_string()
    credentials.check_api(model_name, test_uid, test_key, './', update_rc_file=True)

    # Check that the rc file was properly updated
    expected_content = template.format(uid=test_uid, key=test_key)
    actual_content = rc_path.read_text()
    rc_path.unlink()
    assert expected_content == actual_content, f'{rc_path} was not updated correctly'
