'''
When update_rc_file is True, the RC file should be:
- updated if it already exists,
- created if it doesn't,
- and for .netrc files, it should ONLY update the set of credentials related to
  the given weather model's API URL.
'''

from pathlib import Path
from platform import system
from RAiDER.models import credentials
from test import random_string


def test_cds():
    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['ERA5'])
    rc_path = rc_path.expanduser()

    template = (
        'url: https://cds.climate.copernicus.eu/api/v2\n'
        'key: {uid}:{key}\n'
    )

    test_uid = random_string()
    test_key = random_string()
    credentials.check_api('ERA5', test_uid, test_key, './', update_rc_file=True)

    expected_content = template.format(uid=test_uid, key=test_key)
    actual_content = rc_path.read_text()
    rc_path.unlink()

    assert (
        expected_content == actual_content,
        f'{rc_path} was not updated correctly'
    )


def test_ecmwf():
    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['HRES'])
    rc_path = rc_path.expanduser()

    template = (
        '{{\n'
        '    "url"   : "https://api.ecmwf.int/v1",\n'
        '    "key"   : "{key}",\n'
        '    "email" : "{uid}"\n'
        '}}\n'
    )

    # Simulate a .ecmwfapirc file
    rc_path.write_text(template.format(
        uid=random_string(),
        key=random_string(),
    ))

    test_uid = random_string()
    test_key = random_string()
    credentials.check_api('HRES', test_uid, test_key, './', update_rc_file=True)

    expected_content = template.format(uid=test_uid, key=test_key)
    actual_content = rc_path.read_text()
    rc_path.unlink()

    assert (
        expected_content == actual_content,
        f'{rc_path} was not updated correctly'
    )


def test_netrc():
    # Check extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['GMAO'])
    rc_path = rc_path.expanduser()

    template = (
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

    # Simulate a .netrc file with multiple sets of credentials.
    # The ones not for urs.earthdata.nasa.gov should NOT be touched
    rc_path.write_text(template.format(
        uid=random_string(),
        key=random_string(),
    ))

    test_uid = random_string()
    test_key = random_string()
    credentials.check_api('GMAO', test_uid, test_key, './', update_rc_file=True)

    # Check the content
    expected_content = template.format(uid=test_uid, key=test_key)
    actual_content = rc_path.read_text()

    # Remove local API file
    rc_path.unlink()

    assert (
        expected_content == actual_content,
        f'{rc_path} was not updated correctly'
    )
