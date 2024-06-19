'''
When update_rc_file is False, the RC file should NOT be modified if it already
exists.
'''
from pathlib import Path
from platform import system
from RAiDER.models import credentials


def test_cds():
    # Get extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    # Get the target rc file's path
    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['ERA5'])
    rc_path = rc_path.expanduser()

    # Write some example text to test for
    rc_path.write_text('dummy')

    # Test creation of .cdsapirc file in current dir
    credentials.check_api('ERA5', None, None, './', update_rc_file=False)

    # Assert the content was unchanged
    content = rc_path.read_text()
    rc_path.unlink()
    assert content == 'dummy', f'{rc_path} was modified'


def test_ecmwf():
    # Get extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    # Get the target rc file's path
    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['HRES'])
    rc_path = rc_path.expanduser()

    # Write some example text to test for
    rc_path.write_text('dummy')

    # Test creation of .ecmwfapirc file
    credentials.check_api('HRES', None, None, './', update_rc_file=False)

    # Assert the content was unchanged
    content = rc_path.read_text()
    rc_path.unlink()
    assert content == 'dummy', f'{rc_path} was modified'


def test_netrc():
    # Get extension for hidden files
    hidden_ext = '_' if system() == "Windows" else '.'

    # Get the target rc file's path
    rc_path = Path('./') / (hidden_ext + credentials.RC_FILENAMES['GMAO'])
    rc_path = rc_path.expanduser()

    # Write some example text to test for
    rc_path.write_text('dummy')

    # Test creation of .netrc file
    credentials.check_api('GMAO', None, None, './', update_rc_file=False)

    # Assert the content was unchanged
    content = rc_path.read_text()
    rc_path.unlink()
    assert content == 'dummy', f'{rc_path} was modified'
